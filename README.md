# Thirukural RAG

Describe a modern-life situation in plain English — *"my coworker keeps taking credit for my work"* — and get back the Thirukural couplets that speak to it, each with a grounded explanation of why it applies.

The Thirukural is a 2,000-year-old Tamil text of 1,330 couplets on ethics, governance, and love. Its language is archaic, so naive semantic search over the couplets themselves performs poorly. The core idea of this project is to **enrich every kural with LLM-generated "scenario tags"** (modern situations it applies to), index *those* as separate vectors, and retrieve by matching the user's scenario against the tags rather than the couplet text.

## How it works

```
 upstream dataset ──► flatten ──► tag with LLM ──► embed + index ──► serve ──► UI
 (1,330 kurals)      (Phase 2)    (Phases 3-4)      (Phase 6)       FastAPI    React
                                                        │
                                                    eval (Phase 7)
                                                    recall@K vs. gold set
```

1. **Flatten** — merge two upstream JSON sources into one clean record per kural (Tamil couplet, transliteration, English couplet + explanation, book/iyal/adikaram hierarchy, six commentaries).
2. **Tag** — for each kural, an LLM (Gemini Flash-Lite; Groq was used for the initial small-batch validation) generates 2–12 modern scenario phrasings. Output: `pipeline/data/processed/kurals_enriched.json`.
3. **Index** — a local multilingual sentence-transformer embeds **one point per scenario tag** (plus one "couplet" point per kural as a fallback signal) into Qdrant. A user's query only needs to be close to *one* of a kural's tags.
4. **Eval** — a hand-verified gold set of scenarios → expected kurals measures recall@K across three retrieval modes (tag-only, couplet-only, combined). This is the evidence that the tagging effort actually improves retrieval.
5. **Serve** — FastAPI `POST /search` retrieves the top-k kurals and asks Gemini for a short explanation grounded in the retrieved text.
6. **UI** — a single-page React app: one textarea, results as cards.

## Repository layout

```
.
├── backend/        FastAPI service — retrieval + grounded explanation
├── frontend/       React + Vite single-page UI
├── pipeline/       Data pipeline, index build, and retrieval evaluation
│   ├── data/
│   │   ├── raw/         upstream dataset clone  (gitignored — see fetch_dataset.sh)
│   │   └── processed/   flattened + enriched kurals  (committed — LLM output, not cheaply reproducible)
│   ├── scripts/         flatten → tag → index
│   └── eval/            gold set + recall@K + miss diagnostics
├── docker-compose.yml   local Qdrant
└── qdrant_storage/      Qdrant volume  (gitignored — rebuild with build_index.py)
```

## Quickstart

Prerequisites: Python 3.12, Node 20+, Docker.

### 1. Start Qdrant

```bash
docker compose up -d
```

### 2. Build the index

The enriched dataset is already committed, so on a fresh clone you can skip straight to indexing. The embedding model (~1 GB) downloads on first run.

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r pipeline/requirements.txt

cd pipeline
python3 scripts/build_index.py
python3 eval/run_eval.py          # optional: confirm recall@K on the gold set
cd ..
```

> All `pipeline/` scripts use paths relative to `pipeline/` — run them from inside that folder.

### 3. Run the backend

```bash
pip install -r backend/requirements.txt
cp backend/.env.example backend/.env    # then set GEMINI_API_KEY
cd backend
uvicorn main:app --reload --port 8000
```

Interactive docs at http://localhost:8000/docs.

### 4. Run the frontend

```bash
cd frontend
cp .env.example .env                    # VITE_API_BASE defaults to http://localhost:8000
npm install
npm run dev
```

Open http://localhost:5173.

## API

`POST /search`

```json
{ "scenario": "My coworker keeps taking credit for my work and I want to snap at them in the meeting" }
```

Returns `{ "results": [ ... ] }` — each result carries `kural_number`, `book`, `adikaram`, `tamil_couplet`, `transliteration`, `english_couplet`, `english_explanation`, the `matched_tag` that triggered retrieval, its `similarity_score`, and a Gemini-generated `explanation` grounded in that kural.

Also: `GET /` (service info), `GET /health`.

## Regenerating the data from scratch

Only needed if you want to change the tagging prompt or re-derive everything. Steps 2–4 are already done and their output is committed.

```bash
cd pipeline
scripts/fetch_dataset.sh                            # clone upstream into data/raw/
python3 scripts/flatten_dataset.py                  # → data/processed/kurals_flat.json
export GEMINI_API_KEY=...
python3 scripts/generate_tags_full.py               # → data/processed/kurals_enriched.json  (resumable; saves after every kural)
python3 scripts/build_index.py
```

Helpers: `flag_narrow_tags.py` (find and clear low-quality tags for regeneration), `clear_specific_kurals.py` (clear tags for hand-picked kurals), `verify_gold_candidates.py` (inspect kurals before adding them to the gold set), `eval/diagnose_misses.py` and `eval/batch_diagnose_misses.py` (classify eval misses as tagging problems vs. gold-set ambiguity).

`generate_tags.py` is the original Groq small-batch version, kept for reference — set `GROQ_API_KEY` to use it.

## Backend ↔ pipeline contract

The backend reads what `pipeline/scripts/build_index.py` writes. If you change one side, change the other:

| | Value | Set in |
|---|---|---|
| Qdrant collection | `thirukural` | `build_index.py` / `backend/.env` `COLLECTION_NAME` |
| Embedding model | `paraphrase-multilingual-mpnet-base-v2` | `build_index.py` / `backend/.env` `EMBEDDING_MODEL_NAME` |
| Payload schema | `vector_type` ∈ {`scenario_tag`, `couplet`} + kural fields | `build_index.py` → read by `backend/resources/retrieval.py` |

The query embedding model **must** match the index embedding model or similarity scores are meaningless.

## Environment variables

**`backend/.env`** (see `.env.example`): `QDRANT_HOST`, `QDRANT_PORT`, `COLLECTION_NAME`, `EMBEDDING_MODEL_NAME`, `TOP_K`, `SEARCH_LIMIT_RAW`, `GEMINI_API_KEY`, `EXPLAIN_MODEL_NAME`, `LOG_LEVEL`.

**`frontend/.env`**: `VITE_API_BASE`.

**Pipeline scripts** read `GEMINI_API_KEY` / `GROQ_API_KEY` directly from the shell environment (`export`), not from a `.env` file.

## Data source

Kural text, translations, and commentaries come from [vijayanandrp/Thirukkural-Tamil-Dataset](https://github.com/vijayanandrp/Thirukkural-Tamil-Dataset). Scenario tags in `data/processed/kurals_enriched.json` were generated for this project.
