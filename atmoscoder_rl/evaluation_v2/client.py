"""Server-tokenized context budgeting and durable evidence for every transport attempt."""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import time
from urllib.request import Request, urlopen
from uuid import uuid4

import openai

from ..evaluation.client import (ChatModel, Completion, ModelError, PromptTooLongError,
                                 _field, _with_retry, _is_context_length,
                                 load_config, USAGE_KEYS)
from .audit import record_bytes
from .usage import inspect_usage
from ..paths import REPO_ROOT
from ..frozen import json_bytes, sha256

PROTOCOL = 'd-context-v2-20261010'
CONTEXT = 40960


class ProtocolError(ModelError):
    """Deterministic execution-contract failure; never blindly retry."""


class Journal:
    """Immutable, fsynced event files. A start without an end means unknown usage.

    Different UUIDs survive crashes and resumes without replacing earlier events.
    Directory creation is made durable before the first event is acknowledged.
    """
    def __init__(self, path: Path, identity: dict):
        self.path, self.identity = Path(path), identity
        self.path.mkdir(parents=True, exist_ok=True)
        self.session = uuid4().hex

    def write(self, event: str, **data):
        blob = record_bytes(dict(protocol=PROTOCOL, session=self.session,
                               identity=self.identity, event=event, time_ns=time.time_ns(), **data))
        # Write then link, so readers never see a partially written JSON event.
        temporary = self.path / ('.' + uuid4().hex + '.tmp')
        target = self.path / (uuid4().hex + '.json')
        with temporary.open('xb') as handle:
            handle.write(blob); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, target)
        temporary.unlink()
        for folder in (self.path, self.path.parent):
            fd = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)


def output_budget(prompt_tokens: int, cap: int, repair: bool, context: int = CONTEXT) -> int:
    if any(type(x) is not int or x < 1 for x in (prompt_tokens, cap, context)):
        raise ProtocolError('invalid token count, cap or context')
    remaining = context - prompt_tokens
    if remaining < 1 or (not repair and remaining < cap):
        raise ProtocolError(f'prompt leaves insufficient context: {prompt_tokens}+{cap}>{context}')
    return min(cap, remaining) if repair else cap


class BudgetedModel(ChatModel):
    def __init__(self, config: dict, journal: Journal, *, context: int = CONTEXT):
        super().__init__(config['model_id'], api_key=config['api_key'], base_url=config['base_url'],
                         timeout=config.get('timeout', 1800), temperature=config['temperature'],
                         max_tokens=config['max_tokens'], extra_body=config.get('extra_body'),
                         thinking_echo_field=config.get('thinking_echo_field'),
                         retry_status=config.get('retry_status', ()))
        # All retries must pass _stream and produce their own durable audit events.
        self.client = self.client.with_options(max_retries=0)
        self.tokenize_url = config['base_url'].rstrip('/').removesuffix('/v1') + '/tokenize'
        self.tokenize_key = config['api_key']
        self.context, self.journal = context, journal
        self.content_try = 0
        self.last_call = {}
        self.reported_usage = dict.fromkeys(USAGE_KEYS, 0)
        self.usage_complete = True
        self.usage_unknown = set()

    def remember_usage(self, usage):
        evidence=inspect_usage(usage.model_dump() if usage is not None else None)
        self.usage_complete &= evidence.complete
        for key,value in evidence.counts.items():
            if value is None:
                self.usage_unknown.add(key)
            else:
                self.reported_usage[key] += value
        # A lower bound can include a known prompt even if output count is unknown.
        if evidence.counts['total_tokens'] is None:
            self.reported_usage['total_tokens'] += sum(evidence.counts[k] or 0
                                                      for k in ('prompt_tokens','completion_tokens'))
        return evidence

    def tokenize(self, request: dict) -> dict:
        body = dict(model=request['model'], messages=request['messages'],
                    add_generation_prompt=True, add_special_tokens=False,
                    chat_template_kwargs=(request.get('extra_body') or {}).get('chat_template_kwargs', {}))
        req = Request(self.tokenize_url, data=json_bytes(body),
                      headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.tokenize_key})
        with urlopen(req, timeout=60) as response:
            result = json.load(response)
        tokens, count = result['tokens'], result['count']
        if (type(count) is not int or not isinstance(tokens, list) or count != len(tokens)
                or not all(type(t) is int for t in tokens) or result['max_model_len'] != self.context):
            raise ProtocolError('server tokenizer response or context limit mismatch')
        return dict(prompt_tokens=count, context=self.context, tokens_sha256=sha256(json_bytes(tokens)))

    def _request(self, prompt, system):
        request = super()._request(prompt, system)
        tokenization_id = uuid4().hex
        self.journal.write('tokenize_start', call_id=tokenization_id, request=request,
                           content_try=self.content_try)
        try:
            budget = self.tokenize(request)
            request['max_tokens'] = output_budget(budget['prompt_tokens'], self.max_tokens,
                                                   self.content_try > 0, self.context)
        except Exception as error:
            self.journal.write('tokenize_error', call_id=tokenization_id,
                               error=type(error).__name__ + ': ' + str(error), generation_started=False)
            raise ProtocolError('tokenization/budget preflight failed: ' + str(error)) from error
        self.budget = dict(budget, mode_cap=self.max_tokens, max_tokens=request['max_tokens'],
                           repair=self.content_try > 0, tokenization_id=tokenization_id)
        self.journal.write('tokenize_complete', call_id=tokenization_id, budget=self.budget)
        return request

    def _stream(self, request):
        call_id = uuid4().hex
        content, reasoning = [], []
        usage = None
        finish_reason = None
        self.journal.write('request_start', call_id=call_id, content_try=self.content_try,
                           request=request, budget=self.budget)
        try:
            with self.client.chat.completions.create(
                    **request, stream=True, stream_options={'include_usage': True}) as stream:
                for chunk in stream:
                    if chunk.usage:
                        usage = chunk.usage
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    if choice.finish_reason is not None:
                        finish_reason = choice.finish_reason
                    if choice.delta.content:
                        content.append(choice.delta.content)
                    if self.thinking_echo_field:
                        piece = _field(choice.delta, self.thinking_echo_field)
                        if piece:
                            reasoning.append(piece)
        except Exception as error:
            self.usage_complete = False
            self.remember_usage(usage)
            self.journal.write('request_error', call_id=call_id, error=type(error).__name__ + ': ' + str(error),
                               content=''.join(content), reasoning=''.join(reasoning),
                               provider_usage=usage.model_dump() if usage else None,
                               usage_complete=False, finish_reason=finish_reason)
            raise
        evidence=self.remember_usage(usage)
        answer, thoughts = ''.join(content), ''.join(reasoning)
        completion = Completion(text=answer or thoughts, content=answer, reasoning=thoughts or None,
                                prompt_tokens=evidence.counts["prompt_tokens"] or 0,
                                completion_tokens=evidence.counts["completion_tokens"] or 0,
                                reasoning_tokens=evidence.counts["reasoning_tokens"] or 0)
        self.last_call = dict(call_id=call_id, budget=self.budget, finish_reason=finish_reason,
                              usage_complete=evidence.complete, usage_issues=evidence.issues)
        # Preserve the answer BEFORE validating the protocol or running user-generated code.
        self.journal.write('request_complete', **self.last_call, completion=asdict(completion),
                           provider_usage=usage.model_dump() if usage else None)
        if (not evidence.complete or finish_reason not in {'stop', 'length'}
                or completion.prompt_tokens != self.budget['prompt_tokens']
                or completion.completion_tokens > request['max_tokens']):
            raise ProtocolError('generation usage/finish reason does not match tokenization contract')
        return completion

    def generate(self, prompt, system=None):
        request = self._request(prompt, system)
        try:
            return _with_retry(lambda: self._stream(request), self.retryable_status)
        except ProtocolError:
            raise
        except openai.BadRequestError as error:
            raise (PromptTooLongError if _is_context_length(error) else ModelError)(str(error)) from error
        except Exception as error:
            raise ModelError(str(error)) from error


def build_model(section: str, journal: Journal) -> BudgetedModel:
    return BudgetedModel(load_config(section, REPO_ROOT / "models_v2.toml"), journal)
