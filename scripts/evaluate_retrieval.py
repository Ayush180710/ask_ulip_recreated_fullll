"""Evaluate retrieval and corpus-bound refusal behaviour on a labelled test set."""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.rag import RAGPipeline

DEFAULT_SET = ROOT / 'evaluation' / 'evaluation_set.json'
OUT = ROOT / 'evaluation' / 'retrieval_results.csv'

def evaluate(test_set=DEFAULT_SET, top_k=5):
    cases = json.loads(Path(test_set).read_text(encoding='utf-8'))
    rag = RAGPipeline()
    rows, answerable, hit_at_1, hit_at_k, out_of_corpus, refusals = [], 0, 0, 0, 0, 0
    for case in cases:
        results = rag.retrieve(case['question'], top_k)
        titles = [item['chunk'].title for item in results]
        top1_score = results[0]['score'] if results else None
        top1_rerank = results[0]['rerank_score'] if results else None
        expected = set(case['expected_titles'])
        top1_hit = bool(expected and titles and titles[0] in expected)
        topk_hit = bool(expected.intersection(titles))
        if case['answerable']:
            answerable += 1
            hit_at_1 += top1_hit
            hit_at_k += topk_hit
        else:
            out_of_corpus += 1
            answer, _ = rag.ask(case['question'], top_k)
            normalized_answer = answer.lower()
            refused = any(phrase in normalized_answer for phrase in (
                'do not have enough evidence',
                'not enough evidence',
                'not available in the indexed public corpus',
            ))
            refusals += refused
        rows.append({
            'id': case['id'], 'question': case['question'], 'answerable': case['answerable'],
            'expected_evidence': ' | '.join(case['expected_titles']),
            'top1': titles[0] if titles else '',
            'top1_score': round(top1_score, 4) if top1_score is not None else '',
            'top1_rerank_score': round(top1_rerank, 4) if top1_rerank is not None else '',
            'retrieved_titles': ' | '.join(titles),
            'hit_at_1': top1_hit if case['answerable'] else 'N/A',
            'hit_at_k': topk_hit if case['answerable'] else 'N/A',
            'refusal_correct': refused if not case['answerable'] else 'N/A',
            'manual_answer_faithfulness': ''
        })
    with OUT.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    metrics = {
        'total_cases': len(cases), 'answerable_cases': answerable,
        'out_of_corpus_cases': out_of_corpus,
        'hit_rate_at_1': round(hit_at_1 / answerable, 3) if answerable else 0,
        f'hit_rate_at_{top_k}': round(hit_at_k / answerable, 3) if answerable else 0,
        'out_of_corpus_refusal_rate': round(refusals / out_of_corpus, 3) if out_of_corpus else 0
    }
    (ROOT / 'evaluation' / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    return metrics

if __name__ == '__main__':
    metrics = evaluate(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SET)
    print(json.dumps(metrics, indent=2))
    print(f'Detailed results: {OUT}')
