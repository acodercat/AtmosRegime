"""Validate provider usage without turning missing measurements into zero."""
from dataclasses import dataclass


@dataclass
class UsageEvidence:
    counts: dict
    issues: list[str]

    @property
    def complete(self):
        return not self.issues


def inspect_usage(raw):
    issues=[]
    data=raw if isinstance(raw,dict) else {}
    counts={}
    for key in ('prompt_tokens','completion_tokens'):
        value=data.get(key)
        if type(value) is not int or value<0:
            issues.append('missing or invalid '+key)
            value=None
        counts[key]=value
    p,c=counts['prompt_tokens'],counts['completion_tokens']
    counts['total_tokens']=p+c if p is not None and c is not None else None
    if 'total_tokens' in data:
        total=data['total_tokens']
        if type(total) is not int or total<0 or total!=counts['total_tokens']:
            issues.append('invalid or inconsistent total_tokens')
    details=data.get('completion_tokens_details')
    if details is not None and not isinstance(details,dict):
        issues.append('invalid completion_tokens_details')
        details={}
    reasoning=(details or {}).get('reasoning_tokens')
    if reasoning is None:
        reasoning=data.get('reasoning_tokens')
    # The provider often omits the reasoning share. Retain the legacy zero sentinel,
    # explicitly described as unreported in audit output; not a measurement of zero.
    if reasoning is None:
        reasoning=0
    elif type(reasoning) is not int or reasoning<0 or (c is not None and reasoning>c):
        issues.append('invalid reasoning_tokens')
        reasoning=None
    counts['reasoning_tokens']=reasoning
    return UsageEvidence(counts,issues)
