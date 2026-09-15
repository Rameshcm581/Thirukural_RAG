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

## Deployment (free tier)

The backend loads a ~1 GB embedding model at startup, so it needs 2–3 GB of RAM — more than most free web hosts give you. This stack runs the whole thing for $0:

| Piece | Host | Why |
|---|---|---|
| Vector DB | [Qdrant Cloud](https://cloud.qdrant.io) free cluster (1 GB) | The index is ~40 MB; fits easily |
| Backend | [Hugging Face Spaces](https://huggingface.co/spaces) — Docker SDK, free tier (2 vCPU / 16 GB) | Only free host with enough RAM for the model |
| Frontend | [Vercel](https://vercel.com) Hobby | Static Vite build, root directory `frontend/` |
| LLM | Gemini free tier | Already used for explanations |

### 1. Qdrant Cloud — host the index

1. Sign up at cloud.qdrant.io → **Create cluster** → pick the **Free** tier and a region.
2. Copy the cluster **URL** (`https://….cloud.qdrant.io:6333`) and create an **API key**.
3. Push the index from your machine (re-embeds locally, takes a few minutes):

   ```bash
   cd pipeline
   QDRANT_URL="https://….cloud.qdrant.io:6333" QDRANT_API_KEY="…" python3 scripts/build_index.py
   ```

### 2. Hugging Face Space — host the backend

`backend/` is pushed to the Space as its root; `backend/Dockerfile` and the YAML header in `backend/README.md` are what HF reads.

1. On huggingface.co: **New Space** → name it (e.g. `thirukural-rag-api`) → SDK **Docker** → **Blank** → Public → Create.
2. In the Space's **Settings → Variables and secrets**, add these **Secrets**:
   - `QDRANT_URL` — the cluster URL from step 1
   - `QDRANT_API_KEY`
   - `GEMINI_API_KEY`

   Everything else (`COLLECTION_NAME`, `EMBEDDING_MODEL_NAME`, `TOP_K`, …) has the right default in `backend/configs/base_config.py`.
3. Push just the `backend/` folder to the Space:

   ```bash
   pip install huggingface_hub
   huggingface-cli login            # paste a WRITE token; answer Y to "add as git credential"

   git remote add hf https://huggingface.co/spaces/<HF_USERNAME>/thirukural-rag-api
   git push --force hf "$(git subtree split --prefix backend main)":main
   ```

   The first build takes ~5–10 min (installs CPU torch, bakes the model into the image). Watch it under the Space's **Logs** tab. Re-run the same `git push` line after any backend change.
4. Your API is live at `https://<HF_USERNAME>-thirukural-rag-api.hf.space` — open `/docs` to test.

> Free Spaces go to sleep after 48 h without traffic and take ~1 min to wake. The frontend pings `/health` on page load to start that early.

### 3. Vercel — host the frontend

1. vercel.com → **Add New → Project** → import the GitHub repo.
2. **Root Directory:** `frontend`. Framework is auto-detected as Vite.
3. **Environment Variables:** `VITE_API_BASE` = `https://<HF_USERNAME>-thirukural-rag-api.hf.space` (no trailing slash).
4. **Deploy.** Every push to `main` redeploys automatically.

### Running the backend image locally

```bash
cd backend
docker build -t thirukural-api .
docker run --rm --network host --env-file .env thirukural-api
```

(`--network host` lets the container reach your local Qdrant on `localhost:6333`.)

## Backend ↔ pipeline contract

The backend reads what `pipeline/scripts/build_index.py` writes. If you change one side, change the other:

| | Value | Set in |
|---|---|---|
| Qdrant collection | `thirukural` | `build_index.py` / `backend/.env` `COLLECTION_NAME` |
| Embedding model | `paraphrase-multilingual-mpnet-base-v2` | `build_index.py` / `backend/.env` `EMBEDDING_MODEL_NAME` |
| Payload schema | `vector_type` ∈ {`scenario_tag`, `couplet`} + kural fields | `build_index.py` → read by `backend/resources/retrieval.py` |

The query embedding model **must** match the index embedding model or similarity scores are meaningless.

## Environment variables

**`backend/.env`** (see `.env.example`): `QDRANT_URL` + `QDRANT_API_KEY` (hosted) or `QDRANT_HOST` + `QDRANT_PORT` (local), `COLLECTION_NAME`, `EMBEDDING_MODEL_NAME`, `TOP_K`, `SEARCH_LIMIT_RAW`, `GEMINI_API_KEY`, `EXPLAIN_MODEL_NAME`, `LOG_LEVEL`.

**`frontend/.env`**: `VITE_API_BASE`.

**Pipeline scripts** read `GEMINI_API_KEY` / `GROQ_API_KEY` directly from the shell environment (`export`), not from a `.env` file.

## Data source

Kural text, translations, and commentaries come from [vijayanandrp/Thirukkural-Tamil-Dataset](https://github.com/vijayanandrp/Thirukkural-Tamil-Dataset). Scenario tags in `data/processed/kurals_enriched.json` were generated for this project.
