# Results summary

## Leaderboard
| model_id | scope | n_test | accuracy | macro_f1 | ci_low | ci_high | ece | n_params | latency_ms |
|---|---|---|---|---|---|---|---|---|---|
| llm_groq_zs | llm-subset | 240 | 0.9792 | 0.9791 | 0.9537 | 0.9959 | 0.0807 | — | — |
| muril_multi | full | 940 | 0.9319 | 0.9319 | 0.9009 | 0.9582 | 0.0581 | 237557762.0000 | 13.9600 |
| tfidf_lr | full | 940 | 0.8277 | 0.8277 | 0.7818 | 0.8655 | 0.0606 | 205379.0000 | 1.0472 |
| tfidf_nb | full | 940 | 0.8128 | 0.8128 | 0.7666 | 0.8545 | 0.0509 | 410758.0000 | 3.6576 |
| bilstm_subword | full | 940 | 0.7989 | 0.7988 | 0.7580 | 0.8301 | 0.1685 | 1292674.0000 | 4.3572 |
| tfidf_bnb | full | 940 | 0.7596 | 0.7584 | 0.7065 | 0.8056 | 0.2387 | 410758.0000 | 12.1800 |

## McNemar (best vs others, full test)
| model_a | model_b | n | a_right_b_wrong | a_wrong_b_right | p_value | significant_at_0.05 |
|---|---|---|---|---|---|---|
| muril_multi | tfidf_lr | 940 | 120 | 22 | 0.0000 | True |
| muril_multi | tfidf_nb | 940 | 133 | 21 | 0.0000 | True |
| muril_multi | bilstm_subword | 940 | 152 | 27 | 0.0000 | True |
| muril_multi | tfidf_bnb | 940 | 188 | 26 | 0.0000 | True |

## Fair comparison on the LLM subset
| model_id | n | accuracy | macro_f1 |
|---|---|---|---|
| llm_groq_zs | 240 | 0.9792 | 0.9791 |
| muril_multi | 240 | 0.9167 | 0.9165 |
| tfidf_lr | 240 | 0.7792 | 0.7785 |
| tfidf_nb | 240 | 0.7750 | 0.7747 |
| bilstm_subword | 240 | 0.7417 | 0.7408 |
| tfidf_bnb | 240 | 0.7250 | 0.7241 |
