# Multilingual Sentiment Analysis for Indian-Language Reviews

SMAI Assignment 3 · Theme T8 · Experiment report

## Abstract

We implement a local review analyzer for Hindi, Tamil, Telugu, Bengali, Marathi, and code-mixed Hinglish. The system returns binary sentiment and confidence for a single review, summarizes uploaded CSVs, and exposes per-language evaluation results. The experiment compares classical TF-IDF classifiers, recurrent networks (RNN, LSTM, BiLSTM) and a multilingual transformer (MuRIL) with a prompted LLM. Because IndicSentiment contains translations of the same English reviews, we assign all translations of each source review to a common partition. Evaluation reports macro-F1, accuracy, calibration, confidence intervals, and matched-model comparisons. The system also supports occlusion explanations, theme extraction, and aspect sentiment. On the full 940-review test split, the multilingual MuRIL transformer is the strongest local model (macro-F1 0.932), ahead of TF-IDF logistic regression (0.828). A zero-shot LLM reaches 0.979 on a 240-review subset.

## 1. Introduction and related work

A business collecting reviews across several languages needs aggregate feedback and the ability to inspect an individual decision. The implementation offers a React dashboard and a FastAPI service with a shared model registry, so the batch and single-review interfaces use the same prediction logic. Local models can score review text without sending it to an external LLM provider. When an LLM provider is enabled, it also supports zero-shot sentiment, aspect extraction, and batch themes.

MuRIL is a BERT model pretrained by Google on 17 Indian languages, including transliterated (Roman-script) text. It lets us compare transfer learning with supervised recurrent models and a sparse lexical baseline. [MuRIL model card](https://huggingface.co/google/muril-base-cased)

## 2. Data and leakage prevention

We downloaded the validation and test JSON Lines files for five languages from IndicSentiment, for 5,780 source rows in total. Each language contributes 156 validation rows and 1,000 test rows to the pool. The source contains parallel translations of English reviews with polarity and aspect annotations. The application accepts restaurant, movie, book, or other review text, but the evaluation corpus is the shared translated review dataset. Separate native-domain benchmarks would be needed to measure performance for those individual domains. [IndicSentiment dataset card](https://huggingface.co/datasets/ai4bharat/IndicSentiment)

Preparation pools the source splits, retains valid binary labels, normalizes Unicode, removes URLs, and drops duplicate texts within each language. A normalized English source text determines the group identifier. The split samples groups with stratification and a fixed seed, targeting 70% train, 15% validation, and 15% test. The leakage check verifies that each group appears in exactly one split. Vocabulary construction, document-frequency estimation, and the optional local SentencePiece tokenizer all use the training partition.

The Hinglish procedure samples source reviews from all three partitions, rewrites them into Roman-script Hindi-English, and labels them in a separate LLM call. Rewrites inherit the original group and partition. Held-out rewrites use the source's gold label. The audit reports provider label coverage, agreement, and Cohen's kappa against those source labels. With Groq gpt-oss-20b, 500 rewrites were produced (350 train, 75 validation, 75 test). The LLM labels were all parseable, agreed with the gold labels on 99% of rows, and gave Cohen's kappa 0.980.

## 3. Models and application

1. **TF-IDF with logistic regression.** A pooled sparse feature space combines character and word features. The training script selects the regularization setting using validation macro-F1.
2. **BiLSTM.** A trained embedding table feeds a packed bidirectional LSTM layer. LSTM gates control the cell update, and the forward and backward representations are concatenated. Masked max and mean pooling summarize valid timesteps before the classification layer.
2b. **Unit III classifiers on the same TF-IDF features.** Multinomial Naive Bayes (smoothing alpha), Bernoulli Naive Bayes (each n-gram only present or absent; alpha) reuse the TF-IDF features. Each hyperparameter is chosen by validation macro-F1, so only the classifier differs from model 1.
3. **MuRIL (all languages).** `google/muril-base-cased`, a BERT model pretrained on 17 Indian languages including transliterated text, is fine-tuned on all 4,380 training reviews with class-weighted cross-entropy (4 epochs, learning rate 3e-5, 128 tokens, frozen input embeddings). It was trained on a GTX 1650 GPU with mixed precision in about 35 minutes.
4. **Prompted LLM (reference, not trained).** The experiment asks the provider for structured polarity and confidence, with zero or three examples. A separate prompt extracts aspect polarities. The theme extraction prompt summarizes recurring concerns and positive feedback in batch reviews.
5. **Explanations and efficiency.** Occlusion removes individual words and measures the change in probability for the predicted class. Training records parameter counts and per-review inference latency. Latency reflects the tested machine, model placement, and input lengths.

## 4. Evaluation

Macro-F1 averages the positive and negative class F1 scores with equal weight. The pipeline records accuracy, per-class precision and recall, confusion matrices, model confidence, and expected calibration error. It computes percentile confidence intervals from 1,000 bootstrap resamples. The resampling unit is the source-review group, so all translations of one review are drawn together. McNemar's exact test compares the discordant correctness counts on matched test rows.

The LLM evaluation uses up to 40 examples per language. Local predictions are filtered to the same IDs for a like-for-like subset comparison. Language-specific models have a separate evaluation scope for their corresponding language.

The comparisons are the classifier on fixed TF-IDF features (logistic regression, Multinomial and Bernoulli Naive Bayes) and the multilingual transformer against the best classical and recurrent models.

## 5. Results and run scope

The full test split has 940 reviews: 865 in the five Indic languages and 75 in Hinglish. The best local model on it is **muril_multi**, with macro-F1 **0.932** (95% CI 0.901–0.958), significantly better than every other local model (McNemar p < 0.001 against logistic regression: 120 reviews only MuRIL got right, 22 only logistic regression got right). It leads in every language: Bengali 0.954, Tamil 0.942, Hindi 0.931, Marathi 0.925, Hinglish 0.920 and Telugu 0.913. An earlier Marathi-only MahaBERT (`l3cube-pune/marathi-bert-v2`, T8.6) reached 0.931 on the same Marathi reviews. MuRIL practically ties it (0.925), so the specialist model was removed and one multilingual model now serves every language. It is also well calibrated (ECE 0.058). The best non-transformer model was **tfidf_lr**, with macro-F1 **0.828** (95% group-bootstrap CI 0.782–0.865). TF-IDF selected C = 4 on validation data. The recurrent models trained for up to 20 epochs with early stopping on validation macro-F1, using SentencePiece fitted to the training reviews. McNemar's test found TF-IDF better than the BiLSTM at the 0.05 level (p = 0.034). A vanilla RNN (0.766) and a one-directional LSTM (0.781) were also trained with the same settings; the ordering RNN < LSTM < BiLSTM held, so only the BiLSTM was kept.

| Model | Scope | Test n | Macro-F1 | 95% CI | Accuracy | ECE |
| --- | --- | --- | --- | --- | --- | --- |
| muril_multi | full | 940 | 0.932 | 0.901–0.958 | 0.932 | 0.058 |
| tfidf_lr | full | 940 | 0.828 | 0.782–0.865 | 0.828 | 0.061 |
| bilstm_subword | full | 940 | 0.799 | 0.758–0.830 | 0.799 | 0.168 |
| tfidf_nb (Unit III) | full | 940 | 0.813 | 0.767–0.854 | 0.813 | 0.051 |
| tfidf_bnb (Unit III) | full | 940 | 0.758 | 0.707–0.806 | 0.760 | 0.239 |
| llm_groq_zs (reference) | LLM subset | 240 | 0.979 | 0.954–0.996 | 0.979 | 0.081 |

Among the Unit III classifiers, Multinomial Naive Bayes (0.813) is statistically indistinguishable from logistic regression (McNemar p = 0.19). Four further classifiers were trained and then removed as not useful: a linear SVM (0.823, p = 0.68 against logistic regression) added nothing over logistic regression; an RBF-kernel SVM (0.822) matched the linear SVM but took 431 s to train instead of 12 s, because a linear boundary is already sufficient on sparse n-gram features; K-nearest neighbours (0.799) was weaker, slow at prediction and kept the whole training set in memory; and a decision tree (0.649), which splits on one n-gram at a time, generalized poorly from 4,380 reviews. Bernoulli Naive Bayes (0.758) is clearly worse than the multinomial version (0.813): it discards how strongly each n-gram is weighted and treats each of the roughly 200,000 absent n-grams as weak evidence, so many small "absent" signals can outweigh a few strong words. Its probabilities are pushed to extremes (ECE 0.239).

A manual check revealed a limitation that the test set does not show. MuRIL labels short, casual Hinglish praise such as "Product bahut accha hai" and "Bahut badhiya phone hai, camera ekdum mast" as negative with about 99% confidence, while logistic regression gets the first right. The Hinglish test reviews are long LLM rewrites of English reviews, so they do not represent short colloquial Hinglish. More natural Hinglish training data, such as L3Cube HingCorpus, would be needed to fix this.

The LLM was tested on a 240-review subset, so its score is not directly comparable with the full-split rows. The LLM's three-shot and aspect evaluations were not completed because the free Groq tier reached its daily token limit.

See the [per-language results](../backend/artifacts/results/per_language_macro_f1.csv) and [paired comparisons](../backend/artifacts/results/mcnemar.csv) for the complete outputs.

## 6. Limitations and interpretation

Parallel translated reviews may differ from naturally written reviews. The synthetic Hinglish procedure can preserve the underlying polarity while changing word choice, and a provider may introduce additional label noise. The system predicts a binary polarity and represents mixed opinions through the aspect-level feature. Per-language samples are modest, so the confidence intervals and paired comparisons matter when interpreting small leaderboard gaps. The group bootstrap accounts for correlation among translations of the same source review, so the intervals are wider than a row-level bootstrap would give.

The training split now includes five Indic languages and Hinglish. The local smoke test detected Hinglish for both example inputs from the brief, and TF-IDF labelled both correctly. Hinglish remains the weakest language for the BiLSTM (0.745 macro-F1).

## Acknowledgements

The experiment uses AI4Bharat IndicSentiment and the model configurations published by AI4Bharat and L3Cube. Codex assisted with implementation and verification. The Hinglish data and the zero-shot LLM evaluation used the Groq API.
