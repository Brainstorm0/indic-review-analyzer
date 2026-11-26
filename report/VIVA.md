# Viva preparation

## Dataset and leakage

- **Why group by English source review?** IndicSentiment contains translations of the same review. A row-level random split could place its Hindi translation in training and its Tamil translation in test. `prepare_data.py` puts every translation in the same split and checks group membership after assignment.
- **Where does vocabulary fitting happen?** TF-IDF learns document frequencies from the training set. The recurrent vocabulary and optional local SentencePiece model also fit on the training partition. Validation selects hyperparameters and checkpoints. Test supplies the final measurements.
- **How are Hinglish labels obtained?** One prompt rewrites each source text and a separate prompt assigns polarity. The holdout inherits the source gold label. Cohen’s kappa measures agreement between the provider and gold labels beyond chance.

## Models

- **What changes between RNN, LSTM, and BiLSTM?** The vanilla RNN recalculates a hidden state at each step. The LSTM adds a cell state and gates that control what to retain, update, and expose. BiLSTM reads the sequence in both directions and combines the resulting states.
- **Why pack sequences?** Packing lets the recurrent layer ignore padded positions. Max-plus-mean pooling then captures strong activations and average context over valid time steps. The last-state ablation tests whether that pooling helps.
- **Why clip gradients?** Repeated recurrent operations can produce large gradients. `train_rnn.py` clips the total gradient norm at 1.0 before the optimizer step.
- **Why keep accents in IndicBERT?** Indic characters use combining vowel signs that accent stripping can remove. The slow ALBERT tokenizer is loaded with `keep_accents=True`.
- **How does the frozen probe differ from fine-tuning?** The frozen probe updates only the classification head. Full fine-tuning also changes the pretrained encoder weights. Both use validation macro-F1 for checkpoint selection.

## Evaluation and the app

- **Why macro-F1?** It gives both polarities equal weight. A model that predicts a single majority class can look better in accuracy than in macro-F1.
- **What do the intervals measure?** `evaluation.py` resamples the held-out rows and reports the 2.5th and 97.5th percentiles of the resulting scores. Parallel translations can be correlated, which limits the interpretation of row-level intervals.
- **Why McNemar’s test?** `evaluate_all.py` aligns predictions by test ID, then compares how often each of two models is correct when the other is wrong. The pairing controls for which reviews each model saw.
- **Why rescore on the LLM subset?** Local and LLM results are comparable when calculated on the same review IDs. Whole-test results and language-specific results have different scopes.
- **What does occlusion show?** Each word is temporarily removed and the change in the predicted class probability becomes its importance score. The score can change with the other words around it.
- **How does aspect evaluation count matches?** It matches normalized prediction and gold names by substring, with each item matched at most once. This supports a standard precision, recall, and F1 calculation.

Before presenting, trace a review through `prepare_data.py`, the selected trainer, `evaluation.py`, and the FastAPI prediction endpoint. Use the experiment report and the generated leaderboard when discussing measured results.
