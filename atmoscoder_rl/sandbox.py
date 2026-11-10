"""Isolated execution for rollouts — the part of the reward RL will actually attack.

`engine.run_solver` gives a separate interpreter and a wall-clock timeout, which is what
*evaluation* needs: a frozen model has no incentive to misbehave. A policy under
optimisation does. Two holes in the plain runner were confirmed, not hypothesised:

- **The answer key is readable.** The child inherits the trainer's working directory, so
  `open("data/core.json")`, matched on the problem text, returns the stored
  answers — reward 1.0. Loading the parent's faithful `solve()` the same way also defeats
  input re-sampling. The filesystem and the network are wide open.
- **"Standard library only" is not enforced.** The child sees whatever site-packages the
  trainer's interpreter has. The contract holds in this venv only because nothing else is
  installed; the benchmark's own eval venv has numpy.

Two layers close them, and `isolation()` reports which are active so nothing is assumed:

**Layer 1 — always on, pure Python.** `python -I -S` (no site-packages, no user site, no
`PYTHON*` env: stdlib only, whatever the trainer has installed); a fresh empty temporary
directory as cwd; a scrubbed environment; `resource` limits on address space, file size
(0 — no writes), process count (no fork) and CPU time; and the child in its own session
so a timeout kills the whole process group, not just the direct child.

**Layer 2 — bubblewrap, when it works.** Layer 1 still lets a rollout read any *absolute*
path, and the repo's location is discoverable (`os.walk` from `/`, a few seconds). `bwrap
--unshare-all` with only the interpreter's own directories bound read-only removes the
rest of the filesystem and the network entirely. It is probed once at first use; on
Ubuntu ≥ 23.10 it fails under the default AppArmor policy until
`sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0` (or an AppArmor profile
for bwrap) is set. `atmoscoder_rl.check_reward` refuses to pass while the answer key is
reachable, so a training box cannot silently run without this layer.

**A sandbox that failed to start is not a wrong answer.** Passing the probe once does not
mean every later bwrap starts: on the training boxes bwrap intermittently fails before the
solver runs (`bwrap: Creating new namespace failed: Too many open files`, `bwrap: Failed
to mount tmpfs: ...`), for minutes at a time, and every such call used to come back as
"no output" and score 0 — a correct rollout punished (PREREG_X X-16). Such a failure is
now retried and, if it persists, the solver runs under layers 1 and 1.5 without bwrap.
Either way it is graded on what it computes; `counts()` reports how often each happened.

**A failure counts only if it happens twice (X-24).** On two training boxes, on several
occasions, all reward workers at once scored correct solvers as failed runs for minutes at a
time - 190 of 256 validation answers in one episode, every one of which runs and passes when
re-executed. A genuine failure is deterministic (re-run twice, each of 1,659 failed validation
runs gave the same outcome both times), so a failed run is now run once more and only a second
failure is returned; `counts()` records each recovered first failure by kind. For the
same reason the solver's budget is its CPU time (`RLIMIT_CPU`, timeout + 1 s), not the wall
clock, which on a loaded box or a slow shared filesystem measures the box: the wall-clock limit
(`WALL_TIMEOUT`) is only there to stop a solver that blocks without computing.

**Concurrency is capped, and this is not optional.** One rollout costs a bwrap process
plus an interpreter, and a trainer will ask for as many as it has rollouts. verl runs its
reward loop as several worker processes that each fan out to many concurrent async tasks,
so an uncapped blocking sandbox turns one training step into tens of thousands of
simultaneous process spawns: the first real run put 20,698 bwrap processes on a 224-core
box, 20,745 of them in uninterruptible sleep, drove load average to 19,721 and made the
machine stop accepting ssh. `MAX_CONCURRENCY` (env `ATMOSCODER_SANDBOX_CONCURRENCY`) bounds
it per process — multiply by the number of reward workers to get the machine-wide total,
and keep that at or below the core count.

The protocol is `engine.run_solver`'s, using its `_SOLVER_RUNNER` verbatim, so a solver
that passes under one passes under the other — `check_reward` asserts exactly that.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from functools import lru_cache

from .engine import SOLVER_TIMEOUT, _SOLVER_RUNNER

MEMORY_MB = 512
MAX_PROCESSES = 1        # solve() has no business forking
ENV = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONIOENCODING": "utf-8"}

#: Sandboxes this *process* may run at once. The machine-wide total is this times the
#: number of reward-worker processes, so 4 x verl's 32 workers = 128 on a 224-core box.
#: Deliberately low: the cost of being wrong is a wedged shared machine, and the sandbox
#: spends most of its wall time in process setup rather than on CPU, so a small cap barely
#: dents throughput.
MAX_CONCURRENCY = max(1, int(os.environ.get("ATMOSCODER_SANDBOX_CONCURRENCY", "4")))
_slots = threading.BoundedSemaphore(MAX_CONCURRENCY)

#: bwrap attempts per rollout before falling back to layers 1 and 1.5 (X-16).
BWRAP_ATTEMPTS = 3

#: Wall-clock seconds before a run that is not using its CPU budget is killed (X-24).
WALL_TIMEOUT = 60.0

#: How a child killed for exceeding RLIMIT_CPU ends: by SIGXCPU or SIGKILL directly, or, under
#: bwrap, with the shell convention 128 + signal as bwrap's own exit status.
_CPU_KILLED = {-signal.SIGXCPU, -signal.SIGKILL, 128 + signal.SIGXCPU, 128 + signal.SIGKILL}

_counts: Counter = Counter()
_counts_lock = threading.Lock()

#: The interpreter itself, not the venv symlink to it: with -S no site-packages are
#: wanted, and under bwrap the venv directory is not bound.
PYTHON = os.path.realpath(sys.executable)

#: `ATMOSCODER_SANDBOX_SHARE_NET=1` (PREREG_DOSE_MINISTRAL DM-17): bwrap keeps the host network namespace
#: (`--share-net`) instead of creating one per rollout. On two training hosts creating a network
#: namespace degraded to 10-16 s each after hours of rollouts (kernel rtnl_lock; 0.01 s on a fresh
#: host), which starved training of rewards. Networking stays blocked one layer down: the audit hook
#: refuses every socket / urllib / http.client event, subprocesses and ctypes, and RLIMIT_NPROC is 1,
#: so a solver that tries the network fails exactly as before; the filesystem stays hidden. Off by
#: default; `explore/sandbox_sharenet_check.py` re-grades stored rollouts under both settings.
SHARE_NET = os.environ.get("ATMOSCODER_SANDBOX_SHARE_NET", "0") == "1"

#: Resolved once with the *parent's* PATH. Popen(env=...) resolves a bare argv[0] against
#: the child's PATH, and the scrubbed one does not include ~/.local/bin.
BWRAP = shutil.which("bwrap")

# Resource limits are applied *inside* the sandboxed interpreter, as the first thing it
# does, rather than via preexec_fn on the spawned process: under bubblewrap the spawned
# process is bwrap itself, which must fork before anything runs, and a process cap of 1
# on it would stop the sandbox from starting. The prelude leaves `_SOLVER_RUNNER` verbatim.
_LIMITS_PRELUDE = """
import resource as _r
_r.setrlimit(_r.RLIMIT_AS, ({mem}, {mem}))
_r.setrlimit(_r.RLIMIT_FSIZE, (0, 0))
_r.setrlimit(_r.RLIMIT_NPROC, ({nproc}, {nproc}))
_r.setrlimit(_r.RLIMIT_CPU, ({cpu}, {cpu}))
del _r
"""


#: Installed as the last thing before the untrusted solver is read from stdin and
#: executed. An audit hook cannot be removed once added, and it sits below Python: a
#: rollout cannot reach around it by rebinding `open` or `__import__`.
#:
#: This is layer 1.5, and it exists because layer 2 needs privileges a shared training box
#: often withholds: `bwrap` needs unprivileged user namespaces, which Ubuntu >= 23.10
#: denies by default (`kernel.apparmor_restrict_unprivileged_userns=1`), and without them
#: the plain runner lets a rollout read the answer key by absolute path. The hook closes
#: that hole with no privileges at all. It is weaker than a namespace — it defends the
#: Python level only — so it does not make `bwrap` optional and `isolation()` keeps
#: reporting the two separately.
#:
#: Reads and directory listings are confined to the interpreter's own stdlib (imports must
#: keep working) and the empty temporary cwd. Everything else — the repo, the answer key,
#: /proc, the home directory — raises PermissionError, as do sockets, subprocesses and
#: ctypes (the usual way out of an audit hook).
#:
#: What survives, measured rather than assumed: CPython 3.13 raises no audit event for
#: `os.stat`, so a rollout can still test whether a path exists and how large it is. It
#: cannot open it and cannot list a directory, so the answer key stays unreachable; treat
#: the residue as path *metadata* leakage, and note that `bwrap` removes even that.
_AUDIT_PRELUDE = """
import sys as _s, os as _o, sysconfig as _sc
_allowed = tuple(_o.path.realpath(p) for p in {
    _sc.get_paths().get("stdlib"), _sc.get_paths().get("platstdlib"),
    _o.path.dirname(_o.__file__), _o.getcwd(),
} if p)
_blocked_prefixes = ("socket.", "ctypes.", "urllib.", "ftplib.", "http.client.")
# Escape routes only. `sys._getframe` and `os.putenv` were here and were removed: neither
# reaches the filesystem or the network, both fire constantly inside the stdlib, and
# blocking them can only turn a correct rollout into a zero.
_blocked_events = frozenset((
    "subprocess.Popen", "os.exec", "os.fork", "os.forkpty", "os.spawn", "os.posix_spawn",
    "os.system", "os.rename", "os.remove", "os.rmdir", "os.link", "os.symlink",
    "os.chmod", "os.chown", "os.truncate", "shutil.copyfile", "pty.spawn",
    "webbrowser.open",
))
# "os.stat" is listed for the day CPython raises it; as of 3.13 it does not, which is why
# the docstring claims blocked *reads and listings*, not blocked metadata.
_path_events = frozenset(("open", "os.listdir", "os.scandir", "os.stat", "os.chdir"))

def _guard(event, args):
    if event in _path_events:
        target = args[0] if args else None
        if isinstance(target, int):          # already-open fd (stdin/stdout); not a new read
            return
        if target is None:
            return
        if isinstance(target, bytes):
            target = target.decode("utf-8", "replace")
        try:
            resolved = _o.path.realpath(target)
        except Exception:
            raise PermissionError("sandbox: unresolvable path")
        if not resolved.startswith(_allowed):
            raise PermissionError("sandbox: filesystem access outside the sandbox is blocked")
        return
    if event in _blocked_events or event.startswith(_blocked_prefixes):
        raise PermissionError("sandbox: %s is blocked" % event)

_s.addaudithook(_guard)
del _sc
"""


def _interpreter_args(timeout: float) -> list[str]:
    prelude = _LIMITS_PRELUDE.format(mem=MEMORY_MB * 2**20, nproc=MAX_PROCESSES,
                                     cpu=max(1, int(timeout) + 1))
    return [PYTHON, "-I", "-S", "-c", prelude + _AUDIT_PRELUDE + _SOLVER_RUNNER]


def _bwrap_binds() -> list[str]:
    """Read-only mounts sufficient to start the interpreter and import the stdlib."""
    roots = ["/usr", "/lib", "/lib64", "/bin", "/etc/ld.so.cache", "/etc/alternatives"]
    prefix = sys.base_prefix
    if not prefix.startswith("/usr"):
        roots.append(prefix)
    args: list[str] = []
    for root in roots:
        if os.path.exists(root):
            args += ["--ro-bind", root, root]
    return args


def _bwrap_prefix(cwd: str) -> list[str]:
    args = [BWRAP, "--unshare-all", *(["--share-net"] if SHARE_NET else []), "--die-with-parent", "--new-session",
            "--clearenv"]
    for key, value in ENV.items():
        args += ["--setenv", key, value]
    args += _bwrap_binds()
    args += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
             "--bind", cwd, "/work", "--chdir", "/work", "--"]
    return args


@lru_cache(maxsize=None)
def bwrap_available() -> bool:
    """Probe once: bubblewrap present *and* permitted to create the namespaces it needs."""
    if BWRAP is None:
        return False
    for attempt in range(BWRAP_ATTEMPTS):   # the probe can hit the same transient failure (X-16)
        try:
            with tempfile.TemporaryDirectory() as cwd:
                probe = subprocess.run(
                    _bwrap_prefix(cwd) + [PYTHON, "-I", "-S", "-c", "print('ok')"],
                    capture_output=True, text=True, timeout=15,
                )
        except (OSError, subprocess.TimeoutExpired):
            return False
        if not _bwrap_failed(probe.stdout, probe.stderr):
            break
        _backoff(attempt)
    return probe.returncode == 0 and probe.stdout.strip() == "ok"


def isolation() -> dict[str, bool]:
    """What the current environment can actually enforce. Log this at training start."""
    return {
        "stdlib_only": True,
        "rlimits": True,
        "empty_cwd": True,
        "process_group_kill": True,
        "answer_key_blocked": True,        # audit hook, no privileges needed
        "filesystem_hidden": bwrap_available(),
        "network_hidden": bwrap_available() and not SHARE_NET,
        "network_blocked_by_audit_hook": True,
        "share_net": SHARE_NET,
        "max_concurrency": MAX_CONCURRENCY,
    }


def counts(reset: bool = False) -> dict[str, int]:
    """Calls since start (or the last reset): `bwrap_failed` = bwrap attempts that failed before
    the solver ran, `fallback` = rollouts that then ran without bwrap, one `bwrap_error: <line>`
    count per distinct bwrap error message (`_error_line`), one `fail: <line>` count per kind of
    failed run (`_fail_kind`; X-23), and one `recovered: <line>` count per kind of first failure that
    did not repeat (X-24)."""
    with _counts_lock:
        out = dict(_counts)
        if reset:
            _counts.clear()
    return out


def _count(key: str) -> None:
    with _counts_lock:
        _counts[key] += 1


def _backoff(attempt: int) -> None:
    """Pause before the next bwrap attempt; none after the last one."""
    if attempt + 1 < BWRAP_ATTEMPTS:
        time.sleep(0.1 * (attempt + 1))


def _error_line(stderr: str) -> str:
    """bwrap's own error message, with numbers masked so the count keys stay few."""
    line = stderr.strip().splitlines()[0] if stderr.strip() else ""
    return re.sub(r"\d+", "N", line)[:120]


def _fail_kind(info: str) -> str:
    """The line that names a failed run's cause - the last one, where a traceback puts the exception -
    with numbers and quoted text masked so the count keys stay few."""
    lines = [l for l in info.strip().splitlines() if l.strip()]
    line = lines[-1].strip() if lines else ""
    return re.sub(r"\d+", "N", re.sub(r"(['\"]).*?\1", r"\1…\1", line))[:100]


def _failed(info: str) -> tuple[bool, dict, str]:
    """PREREG_X X-23: count the failure by kind, so the reward log tells a sandbox fault (start
    failure, timeout) from a broken solver. Recording only: the result is returned unchanged."""
    _count("fail: " + _fail_kind(info))
    return False, {}, info


def _bwrap_failed(stdout: str, stderr: str) -> bool:
    """bwrap itself failed, so the solver never ran. bwrap prefixes its own errors with
    "bwrap:"; a solver that runs reports through stdout, even when it raises."""
    return not stdout.strip() and stderr.lstrip().startswith("bwrap:")


def _execute(argv: list[str], cwd: str, code: str, timeout: float) -> tuple[str, str] | str:
    """(stdout, stderr), or the info string of a sandbox that could not start or timed out."""
    try:
        proc = subprocess.Popen(
            argv, cwd=cwd, env=ENV, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, start_new_session=True,
        )
    except OSError as exc:
        return f"Sandbox failed to start: {exc}"
    try:
        stdout, stderr = proc.communicate(code, timeout=max(timeout, WALL_TIMEOUT))
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        return f"Execution timeout (> {max(timeout, WALL_TIMEOUT):g}s wall clock) — blocked"
    if proc.returncode in _CPU_KILLED and not stdout.strip():
        return f"Execution timeout (> {timeout:g}s) — likely an infinite loop"
    return stdout, stderr


def run_solver_sandboxed(code: str, timeout: float = SOLVER_TIMEOUT) -> tuple[bool, dict, str]:
    """`engine.run_solver`, hardened. Same return contract: (ok, results, info).

    Blocks until a concurrency slot is free (see `MAX_CONCURRENCY`). Callers are expected
    to be worker threads or async tasks that can afford to wait; the alternative is
    letting the caller's fan-out become the machine's process count. A failed run is run
    once more, and only a second failure is returned (X-24).
    """
    first = _run_once(code, timeout)
    if first[0]:
        return first
    second = _run_once(code, timeout)
    if second[0]:
        _count("recovered: " + _fail_kind(first[2]))
        return second
    return _failed(second[2])


def _run_once(code: str, timeout: float) -> tuple[bool, dict, str]:
    with _slots, tempfile.TemporaryDirectory(prefix="atmoscoder-rollout-") as cwd:
        argv = _interpreter_args(timeout)
        got = None
        if bwrap_available():
            for attempt in range(BWRAP_ATTEMPTS):
                got = _execute(_bwrap_prefix(cwd) + argv, cwd, code, timeout)
                if isinstance(got, str) or not _bwrap_failed(*got):
                    break
                _count("bwrap_failed")
                _count("bwrap_error: " + _error_line(got[1]))
                got = None
                _backoff(attempt)
            else:
                _count("fallback")
        if got is None:
            got = _execute(argv, cwd, code, timeout)

    if isinstance(got, str):
        return False, {}, got
    stdout, stderr = got
    out = stdout.strip()
    if not out:
        return False, {}, f"Compile/exec error (no output):\n{stderr.strip()[:1000]}"
    try:
        payload = json.loads(out.splitlines()[-1])
    except json.JSONDecodeError:
        return False, {}, f"Unparseable solver output:\n{out[:1000]}"
    if not payload.get("ok"):
        return False, {}, payload.get("err", "unknown solver error")
    return True, payload["result"], "ok"


def _kill_group(proc: subprocess.Popen) -> None:
    """Kill the child and everything it spawned, and do not return until it is reaped.

    `start_new_session` made the child a group leader, so one SIGKILL reaches the whole
    tree. The wait at the end is unconditional on purpose. A child stuck in an
    uninterruptible kernel call (bwrap setting up namespaces on a slow network filesystem
    shows up as `bwrap` in state D) only dies when that call returns; returning early
    released its concurrency slot while it still existed, the next sandbox took the slot
    and stuck the same way, and on 2026-09-08 a cap of 64 became 2,980 live sandboxes and
    a load of 4,400. Holding the slot turns that into a stalled worker thread, which the
    training watchdog can see; releasing it turned it into a wedged machine.
    """
    try:
        os.killpg(proc.pid, 9)
    except ProcessLookupError:
        pass
    try:
        proc.communicate(timeout=5)
        return
    except (subprocess.TimeoutExpired, ValueError):
        pass
    proc.kill()
    for stream in (proc.stdin, proc.stdout, proc.stderr):
        try:
            if stream:
                stream.close()
        except OSError:
            pass
    proc.wait()
