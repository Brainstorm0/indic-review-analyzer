# Expected vs actual

| model | scope | expected_macro_f1 | actual_macro_f1 | status | action |
|---|---|---|---|---|---|
| llm_groq_zs | llm-subset | 0.80-0.98 | 0.9791 | PASS |  |
| muril_multi | full | 0.78-0.96 | 0.9319 | PASS |  |
| tfidf_lr | full | 0.70-0.90 | 0.8277 | PASS |  |
| tfidf_nb | full | 0.65-0.90 | 0.8128 | PASS |  |
| bilstm_subword | full | 0.62-0.88 | 0.7988 | PASS |  |
| tfidf_bnb | full | 0.60-0.88 | 0.7584 | PASS |  |

## Qualitative expectations
| expected behaviour | observed |
|---|---|
| Pretrained MuRIL >= best RNN-family model | YES |
