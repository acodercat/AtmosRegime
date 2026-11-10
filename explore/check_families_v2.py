"""Offline boundary/regression tests; no model requests or benchmark answers."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from atmoscoder_rl.evaluation.client import Completion, ModelError, USAGE_KEYS
from atmoscoder_rl.evaluation.episode import evaluate as old_evaluate, _attempt, REPAIR_PROMPT
from atmoscoder_rl.evaluation_v2.client import BudgetedModel, Journal, ProtocolError, output_budget
from atmoscoder_rl.evaluation_v2.episode import evaluate
from atmoscoder_rl.evaluation_v2.runner import Store, run_job
from atmoscoder_rl.evaluation_v2.audit import summarize, verify_records
from atmoscoder_rl.runs import is_gradable

PROBLEM = dict(id='synthetic', problem='Return the dimensionless number 42.',
               sub_answers=[dict(sub='1', value='42', unit='')])
GOOD = '```python\ndef solve():\n return {"1":{"value":42,"unit":""}}\n```'
CONFIG = dict(model_id='synthetic', api_key='EMPTY', base_url='http://127.0.0.1:9/v1',
              temperature=.3, max_tokens=28672, thinking_echo_field='reasoning',
              extra_body=dict(chat_template_kwargs=dict(enable_thinking=True)))


class FakeModel:
    def __init__(self, outcomes, journal):
        self.outcomes = iter(outcomes)
        self.journal = journal
        self.reported_usage = dict.fromkeys(USAGE_KEYS, 0)
        self.usage_complete = True
        self.last_call = {}
    def generate(self, *args):
        item = next(self.outcomes)
        if isinstance(item, Exception):
            raise item
        result = Completion(text=item, content=item, prompt_tokens=20, completion_tokens=10)
        for key,value in result.usage.items():
            self.reported_usage[key] += value
        return result


class Checks(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name)
        self.journal = Journal(self.path/'audit', {'problem':'synthetic'})
    def tearDown(self):
        self.tmp.cleanup()
    def test_budget(self):
        self.assertEqual(output_budget(100,28672,False),28672)
        self.assertEqual(output_budget(100,28672,True),28672)
        self.assertEqual(output_budget(15587,28672,True),25373)
        self.assertEqual(output_budget(40959,28672,True),1)
        for n,repair in [(12289,False),(40960,True),(41000,True),(0,True)]:
            with self.subTest(n=n,repair=repair), self.assertRaises(ProtocolError):
                output_budget(n,28672,repair)
    def test_correct_and_wrong_equal_v1(self):
        for text in (GOOD, GOOD.replace('42','420')):
            a = old_evaluate(PROBLEM,FakeModel([text],self.journal),.05)
            b = evaluate(PROBLEM,FakeModel([text],self.journal),.05)
            for key in ('passed','details','num_attempts','attempts','usage'):
                self.assertEqual(a[key],b[key])
    def test_repair_equal_v1(self):
        replies=['not python',GOOD]
        a=old_evaluate(PROBLEM,FakeModel(replies,self.journal),.05)
        b=evaluate(PROBLEM,FakeModel(replies,self.journal),.05)
        for key in ('passed','details','num_attempts','attempts','usage'):
            self.assertEqual(a[key],b[key])
    def test_five_tries_and_wrong_no_retry(self):
        record=evaluate(PROBLEM,FakeModel(['not python']*5,self.journal),.05)
        self.assertEqual(record['num_attempts'],5)
        self.assertFalse(record['passed'])
        self.assertTrue(is_gradable(record))
        record=evaluate(PROBLEM,FakeModel([GOOD.replace('42','420')],self.journal),.05)
        self.assertEqual(record['num_attempts'],1)
        self.assertFalse(record['passed'])
    def test_partial_error_excluded_usage_retained(self):
        record=evaluate(PROBLEM,FakeModel(['not python',ProtocolError('overflow')],self.journal),.05)
        self.assertFalse(is_gradable(record))
        self.assertEqual(record['completed_calls_before_error'],1)
        self.assertEqual(record['usage']['total_tokens'],30)
        self.assertEqual(len(record['partial_attempts']),1)
        self.assertEqual(record['attempts'],[])
        self.assertFalse(record['usage_complete'])
    def test_first_error_and_blind_retries(self):
        record=evaluate(PROBLEM,FakeModel([ModelError('down')]*3,self.journal),.05)
        self.assertFalse(is_gradable(record))
        self.assertEqual(record['usage']['total_tokens'],0)
        self.assertFalse(record['usage_complete'])
        self.assertTrue(evaluate(PROBLEM,FakeModel([ModelError('down'),GOOD],self.journal),.05)['passed'])
    def test_resume_retains_failure_and_rejects_completed_replace(self):
        path=self.path/'result.json'
        failure=dict(id='x',passed=False,error='transport',partial_attempts=[{'response':'retained'}])
        store=Store(path,{'protocol':'test'});store.add(failure)
        store=Store(path,{'protocol':'test'})
        self.assertEqual(store.done_ids,set())
        store.add(dict(id='x',passed=True))
        self.assertEqual(Store(path,{'protocol':'test'}).superseded,[failure])
        with self.assertRaises(ValueError): store.add(dict(id='x',passed=False))
        with self.assertRaises(ValueError): Store(path,{'protocol':'other'})
        path.write_text('{broken')
        with self.assertRaises(ValueError): Store(path,{'protocol':'test'})
    def test_orphan_and_failed_calls_survive_new_session(self):
        self.journal.write('request_start',call_id='crashed')
        second=Journal(self.path/'audit', {'problem':'synthetic'})
        second.write('request_start',call_id='retry')
        second.write('request_complete',call_id='retry',usage_complete=True,
                     provider_usage=dict(prompt_tokens=3,completion_tokens=4))
        result=summarize(self.path/'audit')
        self.assertEqual(result['requests'],2)
        self.assertEqual(result['uncertain_requests'],1)
        self.assertEqual(result['known_reported_usage']['total_tokens'],7)
        self.assertFalse(result['usage_complete'])
    def test_exact_request_and_unicode_no_truncation(self):
        model=BudgetedModel(CONFIG,self.journal)
        text='长推理 Δ T = 🌊\n'*200
        model.content_try=1
        with patch.object(model,'tokenize',return_value=dict(prompt_tokens=15587,context=40960,tokens_sha256='test')):
            request=model._request(text,'system')
        self.assertEqual(request['messages'][1]['content'],text)
        self.assertEqual(request['max_tokens'],25373)
        self.assertEqual(request['extra_body']['chat_template_kwargs'],{'enable_thinking':True})
        model.client.close()
    def test_stream_failure_retains_partial_text(self):
        model=BudgetedModel(CONFIG,self.journal)
        model.budget=dict(prompt_tokens=10)
        class Stream:
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def __iter__(self):
                yield NS(usage=None,choices=[NS(finish_reason=None,delta=NS(content='partial',reasoning=None))])
                raise ConnectionError('synthetic connection lost')
        with patch.object(model.client.chat.completions,'create',return_value=Stream()):
            with self.assertRaises(ConnectionError): model._stream({'max_tokens':20})
        rows=[json.loads(p.read_text()) for p in (self.path/'audit').glob('*.json')]
        error=next(r for r in rows if r['event']=='request_error')
        self.assertEqual(error['content'],'partial')
        self.assertFalse(error['usage_complete'])
        self.assertFalse(model.usage_complete)
        model.client.close()
    def test_count_mismatch_preserves_response_then_fails(self):
        model=BudgetedModel(CONFIG,self.journal)
        model.budget=dict(prompt_tokens=10)
        class Usage:
            prompt_tokens=11;completion_tokens=2;completion_tokens_details=None
            def model_dump(self): return dict(prompt_tokens=11,completion_tokens=2)
        class Stream:
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def __iter__(self):
                yield NS(usage=Usage(),choices=[NS(finish_reason='stop',delta=NS(content='response',reasoning=None))])
        with patch.object(model.client.chat.completions,'create',return_value=Stream()):
            with self.assertRaises(ProtocolError): model._stream({'max_tokens':20})
        self.assertEqual(model.reported_usage['total_tokens'],13)
        rows=[json.loads(p.read_text()) for p in (self.path/'audit').glob('*.json')]
        self.assertEqual(next(r for r in rows if r['event']=='request_complete')['completion']['text'],'response')
        model.client.close()
    def test_runner_stops_new_work_and_resumes_both_failures_and_missing(self):
        items=[dict(PROBLEM,id=str(i)) for i in range(10)]
        inp=self.path/'input.json'; inp.write_text(json.dumps(items))
        output=self.path/'result.json'
        created=[]
        def fail(section,journal):
            created.append(journal.identity['problem'])
            model=FakeModel([ProtocolError('synthetic')],journal)
            model.client=NS(close=lambda:None)
            return model
        with patch('atmoscoder_rl.evaluation_v2.runner.build_model',side_effect=fail):
            self.assertEqual(run_job('d-v2-qwen35-9b-off',inp,output,'synthetic',1,2),1)
        self.assertEqual(len(created),2)
        first=json.loads(output.read_text())
        self.assertEqual(len(first['results']),2)
        def success(section,journal):
            model=FakeModel([GOOD],journal);model.client=NS(close=lambda:None);return model
        with patch('atmoscoder_rl.evaluation_v2.runner.build_model',side_effect=success):
            self.assertEqual(run_job('d-v2-qwen35-9b-off',inp,output,'synthetic',1,2),0)
        final=json.loads(output.read_text())
        self.assertEqual(len(final['results']),10)
        self.assertEqual(len(final['superseded']),2)
        self.assertEqual(final['metrics']['errors'],0)
    def test_transport_retry_audits_each_call_and_marks_cost_incomplete(self):
        model=BudgetedModel(CONFIG,self.journal)
        self.assertEqual(model.client.max_retries,0)
        class Usage:
            prompt_tokens=10;completion_tokens=2;completion_tokens_details=None
            def model_dump(self): return dict(prompt_tokens=10,completion_tokens=2)
        class Stream:
            def __init__(self,fail): self.fail=fail
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def __iter__(self):
                if self.fail: raise ConnectionError('synthetic retry')
                yield NS(usage=Usage(),choices=[NS(finish_reason='stop',delta=NS(content='response',reasoning=None))])
        with patch.object(model,'tokenize',return_value=dict(prompt_tokens=10,context=40960,tokens_sha256='test')), \
             patch.object(model.client.chat.completions,'create',side_effect=[Stream(True),Stream(False)]), \
             patch('atmoscoder_rl.evaluation.client.time.sleep'):
            result=model.generate('synthetic','system')
        self.assertEqual(result.text,'response')
        evidence=summarize(self.path/'audit')
        self.assertEqual(evidence['requests'],2)
        self.assertEqual(evidence['uncertain_requests'],1)
        self.assertEqual(evidence['known_reported_usage']['total_tokens'],12)
        self.assertFalse(model.usage_complete)
        model.client.close()
    def test_nonfinite_answers_keep_legacy_grading_and_are_durable(self):
        for value in ('float("inf")','float("nan")'):
            code='def solve():\n return {"1":{"value":'+value+',"unit":""}}'
            old=old_evaluate(PROBLEM,FakeModel([code],self.journal),.05)
            new=evaluate(PROBLEM,FakeModel([code],self.journal),.05)
            self.assertEqual(new['passed'],old['passed'])
            store=Store(self.path/'nonfinite.json',{'test':value})
            store.add(new)
            self.assertTrue(json.loads((self.path/'nonfinite.json').read_text())['results'])
            (self.path/'nonfinite.json').unlink()
    def test_long_reasoning_fallback_preserved(self):
        text='Let us think about the synthetic problem. '*2000
        outcome=_attempt(text)
        self.assertEqual(outcome.code,text.strip())
        self.assertIn(text.strip(),REPAIR_PROMPT.format(problem='synthetic',code=outcome.code,error=outcome.error))


if __name__=='__main__':
    unittest.main(verbosity=2)
