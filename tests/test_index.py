import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_labelled_evaluation_set_has_assignment_scale_and_refusal_cases():
    cases = json.loads((ROOT / 'evaluation' / 'evaluation_set.json').read_text(encoding='utf-8'))
    assert 40 <= len(cases) <= 50
    assert sum(case['answerable'] for case in cases) >= 30
    assert sum(not case['answerable'] for case in cases) >= 8
    assert all(case['expected_titles'] for case in cases if case['answerable'])
    assert all(not case['expected_titles'] for case in cases if not case['answerable'])


def test_raw_corpus_has_traceable_public_source_metadata():
    documents = sorted((ROOT / 'data' / 'raw').glob('*.txt'))
    assert len(documents) >= 10
    for document in documents:
        header = document.read_text(encoding='utf-8').split('\n\n', 1)[0]
        assert 'TITLE:' in header
        assert 'SOURCE:' in header
        assert 'URL: https://' in header
