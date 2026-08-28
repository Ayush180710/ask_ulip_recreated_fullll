# Ask ULIP — final report outline

Use this as an 8–12 page report structure. Replace bracketed placeholders after running the evaluation.

1. **Problem and ULIP relevance:** Explain why scattered public logistics information slows onboarding and how evidence-backed answers help stakeholders.
2. **Scope and data policy:** State that the corpus contains 16 public PIB/DPIIT documents, list their URLs from `data/raw/`, and confirm that no confidential data or live ULIP API was used.
3. **Data preparation and EDA:** Report document count, chunk count, chunk-size/overlap settings, document dates and source distribution. Include a small table or chart.
4. **Method:** Diagram chunking → embeddings → vector retrieval → relevance threshold → grounded generation/citations. Explain the offline hashing-vector baseline.
5. **Evaluation design:** Describe the labelled 42-question set: 32 answerable cases with expected evidence and 10 out-of-corpus cases. Define Hit@1, Hit@5, refusal rate and manual faithfulness review.
6. **Results:** Insert values from `evaluation/metrics.json` and error examples from `evaluation/retrieval_results.csv`. State both successes and failures honestly.
7. **Demo walkthrough:** Show an answerable query, its source links, and an out-of-corpus query that is refused.
8. **Limitations:** Small corpus, lexical fallback embeddings, threshold sensitivity, and no automatic semantic faithfulness metric.
9. **Future work:** Compare two embedding models and two chunking strategies; expand only with attributable public sources; add automated citation/faithfulness checks.
10. **Conclusion:** Summarise the stakeholder value and the evidence that the prototype works.
