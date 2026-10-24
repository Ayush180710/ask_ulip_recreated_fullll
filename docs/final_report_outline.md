# Ask ULIP — final report outline

Use this as an 8–12 page report structure. Replace bracketed placeholders after running the evaluation.

1. **Problem and ULIP relevance:** Explain why scattered public logistics information slows onboarding and how evidence-backed answers help stakeholders. Frame the second problem explicitly: a naive RAG system answers questions its corpus cannot support, so knowing when to refuse matters as much as retrieving well.
2. **Scope and data policy:** State that the corpus contains 16 public PIB/DPIIT documents, list their URLs from `data/raw/`, and confirm that no confidential data or live ULIP API was used.
3. **Data preparation and EDA:** Report document count, chunk count, chunk-size/overlap settings, document dates and source distribution. Include a small table or chart.
4. **Method:** Diagram the two-stage pipeline — chunking → bi-encoder embeddings (all-MiniLM-L6-v2) → FAISS IndexFlatIP retrieval of 15 candidates → cross-encoder reranking (ms-marco-MiniLM-L-6-v2) to top 5 → relevance threshold → grounded generation with citations. Explain why a bi-encoder alone scores topical resemblance rather than answerability, and why the cross-encoder runs only over a shortlist.
5. **Evaluation design:** Describe the labelled 42-question set: 32 answerable cases with expected evidence and 10 out-of-corpus cases. Define Hit@1, Hit@5, refusal rate and manual faithfulness review.
6. **Results:** Insert values from `evaluation/metrics.json` and error examples from `evaluation/retrieval_results.csv`. Present the three configurations side by side (bi-encoder at 0.10, bi-encoder at 0.55, bi-encoder plus reranking at 2.0) to show that reranking removed the recall-versus-refusal trade-off. State both successes and failures honestly, including the Hit@1 regression from 0.719 to 0.688.
7. **Threshold tuning:** Report the swept score distributions for both stages — bi-encoder answerable 0.399–0.831 against out-of-corpus 0.098–0.631, overlapping; cross-encoder answerable 2.28–9.94 against out-of-corpus -11.21–2.19, nearly separable. Justify choosing 2.0 over the perfect-scoring 2.24 on overfitting grounds.
8. **Demo walkthrough:** Show an answerable query, its source links, and an out-of-corpus query that is refused.
9. **Limitations:** Small corpus, thresholds tuned on the same set used for reporting, one out-of-corpus case still passing the gate ("What is ULIP's annual profit for 2026?" at 2.19), dense-only retrieval with no lexical component, and no automatic semantic faithfulness metric.
10. **Future work:** Hybrid retrieval combining BM25 with dense vectors for exact figures and rare identifiers; a held-out test split for threshold tuning; automated groundedness checking on generated answers; comparison of chunking strategies.
11. **Conclusion:** Summarise the stakeholder value and the evidence that the prototype works.
