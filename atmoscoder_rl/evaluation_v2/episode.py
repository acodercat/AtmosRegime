"""Identical grading/repair semantics to v1, with explicit infrastructure exclusion."""
from ..evaluation.client import USAGE_KEYS, ModelError, PromptTooLongError
from ..evaluation.episode import (MAX_ATTEMPTS, API_RETRIES, REPAIR_PROMPT, Attempt,
                                  _attempt, _log, _total_usage)
from ..grading import verify
from ..prompts import SYSTEM, user_prompt
from .client import ProtocolError


def evaluate(problem, model, tolerance, max_attempts=MAX_ATTEMPTS):
    prompt = user_prompt(problem['problem'])
    calls = []
    outcome = Attempt(error='no response')
    try:
        while len(calls) < max_attempts:
            model.content_try = len(calls)
            for retry in range(API_RETRIES + 1):
                try:
                    completion = model.generate(prompt, SYSTEM)
                    break
                except (ProtocolError, PromptTooLongError):
                    raise
                except ModelError:
                    if retry == API_RETRIES:
                        raise
            outcome = _attempt(completion.text)
            calls.append(dict(_log(prompt, completion, 'graded' if not outcome.error
                                   else f'ungradable: {outcome.error}'), **model.last_call))
            if not outcome.error:
                break
            prompt = REPAIR_PROMPT.format(problem=problem['problem'],
                                          code=outcome.code or '(no code was produced)', error=outcome.error)
    except ModelError as error:
        # Keep legacy exclusion sentinels. Partial answers are not graded failures.
        record = dict(id=problem['id'], passed=False, num_attempts=0, error=str(error)[:200],
                      error_type=type(error).__name__, attempts=[], partial_attempts=calls,
                      completed_calls_before_error=len(calls), usage_complete=False)
    else:
        record = dict(id=problem['id'], num_attempts=len(calls), attempts=calls,
                      usage_complete=model.usage_complete)
        if outcome.error:
            record.update(passed=False, failed_reason=outcome.error)
        else:
            passed, details = verify(outcome.results, problem['sub_answers'], tolerance, units='benchmark')
            record.update(passed=passed, details=details)
    unknown=getattr(model, 'usage_unknown', set())
    record['usage'] = {k:None if k in unknown else v for k,v in model.reported_usage.items()}
    record['known_reported_usage'] = dict(model.reported_usage)
    record['completed_response_usage'] = _total_usage(calls)
    record['audit_session'] = model.journal.session
    model.journal.write('episode_complete', record=record)
    return record
