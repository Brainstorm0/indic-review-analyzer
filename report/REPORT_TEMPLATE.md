# Multilingual Sentiment Analysis for Indian-Language Reviews: RNNs, Pretrained Transformers and LLMs
(ICVGIP-style, 6–8 pages)
## Abstract (150 words)
## 1 Introduction — problem, small-business motivation, contributions (bullet list)
## 2 Related work — IndicBERT, L3Cube MahaBERT/HingBERT, LLM zero-shot classification, RNN/LSTM baselines
## 3 Data — IndicSentiment (parallel, CC0); cleaning; group-aware split (why leakage matters with parallel data);
   Hinglish generation + LLM labels; table from artifacts/results/data_stats.json; Cohen's kappa audit
## 4 Method — TF-IDF+LR; RNN/LSTM/BiLSTM (equations, pooling); IndicBERT / MarathiBERT fine-tuning; LLM prompting;
   ABSA; theme extraction; occlusion explanations
## 5 Experimental setup — hyper-parameters (from metrics_*.json "hparams"), hardware, seeds, metrics, bootstrap CIs, McNemar
## 6 Results — leaderboard (summary.md), fig_macro_f1.png, per-language heatmap, fair LLM-subset comparison, McNemar table
## 7 Ablation — fig_ablations.png and the 8 ablation groups; discuss each in 2–3 sentences
## 8 Error analysis & calibration — confusion matrices, reliability diagram, 5 misclassified examples per language
## 9 Limitations — translated (not native) reviews, synthetic Hinglish, binary labels, LLM non-determinism, small test per language
## 10 Conclusion
## Acknowledgements — LLMs used (name + purpose), datasets/models licences

## References
