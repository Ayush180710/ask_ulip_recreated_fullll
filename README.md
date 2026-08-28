# Ask ULIP — RAG-based Logistics Knowledge Assistant

This project fulfils **Option 7: “Ask ULIP”** of the ULIP ML/AI trainee assignment. It answers questions about public ULIP and Indian logistics-policy material, displays the evidence used, and refuses questions that have insufficient corpus evidence.

## Problem and users

ULIP onboarding, use cases and policy information are spread across public releases and documents. A logistics stakeholder or new team member needs a quick, evidence-backed explanation without searching every source manually. This demo is a retrieval-augmented generation (RAG) assistant; it is **not** connected to live ULIP services.

## What is included

- 16 curated public PIB/DPIIT documents in `data/raw/`, each retaining title, source URL and date
- A reproducible cleaning, chunking, embedding and vector-index build pipeline
- Streamlit chat demo with public-source links and retrieved evidence
- Gemini generation when `GEMINI_API_KEY` is configured; an offline evidence extract otherwise
- A labelled 42-case evaluation set: 32 answerable questions and 10 deliberately out-of-corpus/hallucination cases
- Retrieval Hit@1 / Hit@5 and refusal-rate evaluation, with a blank reviewer column for answer-faithfulness assessment

## Architecture

Public corpus → cleaning/chunking → embeddings → vector search → relevance guard → grounded answer → clickable source citations

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

Add a Gemini key to `.env` if you want generated answers. Without a key, the app still demonstrates retrieval and source grounding using the local corpus.

## Rebuild the included index

```bash
python scripts/build_index.py
```

## Evaluate retrieval

```bash
python scripts/evaluate_retrieval.py
```

The command writes `evaluation/retrieval_results.csv` and `evaluation/metrics.json`. Review the `manual_answer_faithfulness` column by checking that each answer makes only claims supported by its cited sources. This is deliberately separate from retrieval metrics.

## Method and assumptions

The bundled index uses 4,096-dimensional deterministic hashing embeddings and cosine similarity so that the project runs offline. `MIN_RETRIEVAL_SCORE` (default `0.10`) guards against treating a merely nearest chunk as relevant evidence. It can be tuned using the labelled evaluation set. For a stronger final version, compare this baseline with `sentence-transformers/all-MiniLM-L6-v2` and record the same metrics.

The file name `faiss.index` is retained for compatibility, but the portable bundled index is NumPy-backed, not a FAISS binary. A production implementation can replace it with FAISS/Chroma and semantic embeddings without changing the application flow.

## Corpus policy

The corpus uses public Government of India / PIB / DPIIT material and is intended for demonstration. The internship assignment requires public, synthetic or explicitly provided data and prohibits confidential ULIP/company data or unauthorized live ULIP APIs.

## Assignment hand-in checklist

- Repository: this codebase, README, requirements and sample public corpus
- Evaluation: run the command above and include the generated metrics and reviewed CSV in the report
- Demo: run `streamlit run app.py`; ask one answerable and one out-of-corpus question
- Report/presentation: use [`docs/final_report_outline.md`](docs/final_report_outline.md) to explain the problem, corpus, baseline, results, limitations and next steps
