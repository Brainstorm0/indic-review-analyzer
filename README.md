# Indic Review & Sentiment Analyzer

This SMAI T8 project uses classical TF-IDF classifiers, a BiLSTM network and the MuRIL transformer to classify reviews in Hindi, Tamil, Telugu, Bengali, Marathi, and Hinglish. The React dashboard supports single-review prediction, CSV analysis, aspect sentiment, and comparisons of local and zero-shot models.

## 1. Set up the backend

Use 64-bit Python 3.11 and Node.js 20 or later. From the project folder in PowerShell, run the setup below if `backend/.venv` is missing. An existing workspace environment can run directly through `.venv/Scripts/python.exe`.

```powershell
Set-Location backend
py -3.11 -m venv .venv
$env:PYTHONUTF8 = '1'
$python = '.\.venv\Scripts\python.exe'
& $python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
& $python -m pip install -r requirements.txt
& $python -m scripts.check_env
```

## 2. Configure an LLM provider

Copy `.env.example` to `.env` if it does not already exist. Select Groq (the default), Gemini, or a running local Ollama instance in `LLM_PROVIDER`, and list the providers to evaluate in `LLM_EVAL_PROVIDERS`. Put any API key directly in `.env`. Verify the configured provider with:

```powershell
& $python -m scripts.list_llm_models
```

The provider is used for Hinglish generation and labeling, LLM evaluation, themes, and aspect extraction.
For the local option, start Ollama, run `ollama pull llama3.1:8b`, and set `LLM_PROVIDER=ollama`.

## 3. Train and evaluate

The core models are TF-IDF + logistic regression, BiLSTM and MuRIL. MuRIL (`google/muril-base-cased`) is fine-tuned on all languages and is the app's default model (macro-F1 0.932). Training it takes about 35 minutes on a 4 GB GPU (`backend/.venv-gpu` with the CUDA build of PyTorch) or several hours on a CPU. The Unit III classifiers (Multinomial and Bernoulli Naive Bayes) are trained on the same TF-IDF features for comparison (`python -m scripts.train_tfidf --clf nb|bnb`). A zero-shot Groq LLM serves as a reference model and powers the aspect and theme features.

```powershell
& $python -m scripts.run_pipeline
```

The pipeline downloads IndicSentiment and MuRIL, prepares a group-disjoint split, and generates 500 Hinglish examples with the LLM. It then trains the models, runs the LLM evaluations, and writes metrics and figures under `backend/artifacts`. Use `--quick` for a shorter run or `--skip-llm` when a provider is unavailable. Use `--only` with comma-separated stage names (`prepare,hinglish,tfidf,nb,bnb,bilstm,muril_multi,llm_zs,llm_fs,aspects,evaluate,expected`) to resume a partial run.

The recurrent models fit SentencePiece on the training partition and save the tokenizer alongside each model. MuRIL trains with a frozen input embedding table (`--freeze-embeddings`), which keeps CPU memory use manageable on Windows.

The free Groq tier allows 200k tokens per day. A full run of the Hinglish generation, both LLM evaluations and the aspect evaluation needs more than that, so the LLM stages may have to be resumed the next day with `--only`. Responses are cached in `backend/artifacts/llm_cache`, so finished calls are not repeated.

## 4. Start the app

From `backend`, start the API:

```powershell
& .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

From `frontend` in another terminal, start React:

```powershell
npm.cmd install
npm.cmd run dev
```

Open [the dashboard](http://localhost:5173). Use the sample CSV on the Batch page to try the upload flow. The Research page reads the evaluation artifacts produced in step 3.

For a single local server, run `npm.cmd run build` in `frontend`, then start the API and open [port 8000](http://127.0.0.1:8000).

## 5. Prepare the submission

Use `report/REPORT_TEMPLATE.md` with measured results from `backend/artifacts/results` and the figures from `backend/artifacts/figures`. The pitch and demo recording depend on the trained app and its results. Exclude `backend/.env`, `backend/.venv`, `backend/.venv-gpu`, `frontend/node_modules`, and `backend/artifacts/models` from the submission archive.
