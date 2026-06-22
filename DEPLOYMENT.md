# Deployment Guide

This project is split into two deployable parts:

- **Frontend:** React/Vite/Tailwind dashboard on Vercel.
- **Backend:** FastAPI optimizer on Render using Docker.

The frontend is intentionally lightweight. It does not install Python, Torch, Transformers, LLMLingua, sentence-transformers, SciPy, or scikit-learn.

## Architecture

```text
Browser
  -> Vercel frontend
  -> Render FastAPI backend
  -> Gemini / OpenAI provider APIs
```

## Render Backend

Render should deploy the repository root using `render.yaml` or the root `Dockerfile`.

Required environment variables:

```text
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash
ENABLE_LLM_LINGUA=1
LLM_LINGUA_BACKEND=llmlingua2
LLM_LINGUA2_DEVICE=cpu
DATABASE_PATH=/tmp/app.sqlite3
CORS_ORIGINS=https://your-vercel-app.vercel.app,http://localhost:5173
```

Optional:

```text
OPENAI_API_KEY=...
REQUEST_SIZE_LIMIT_BYTES=1000000
GEMINI_FLASH_INPUT_PER_1K=0.0003
GEMINI_FLASH_OUTPUT_PER_1K=0.0025
```

Render health check:

```text
GET /health
```

Main API endpoints used by the frontend:

```text
GET /health
POST /optimize
POST /v1/chat/completions
POST /benchmark
POST /evaluate-quality
POST /robustness-test
POST /company-pilot-sim
GET /analytics
GET /traces
```

## Vercel Frontend

Recommended Vercel setup:

```text
Framework preset: Vite
Root directory: repository root
Build command: npm --prefix frontend ci && npm --prefix frontend run build
Output directory: frontend/dist
```

Set this Vercel environment variable:

```text
VITE_API_URL=https://your-render-backend.onrender.com
```

The root `.vercelignore` excludes the backend, tests, backups, SQLite files, logs, and model/cache artifacts from Vercel uploads.

## Local Development

Backend:

```powershell
cd "C:\Users\tanis\OneDrive\Documents\New project 2\ai-cost-optimization-middleware"
$env:GEMINI_API_KEY="..."
$env:CORS_ORIGINS="http://127.0.0.1:5173,http://localhost:5173"
uvicorn app.main:app --reload
```

Frontend:

```powershell
cd "C:\Users\tanis\OneDrive\Documents\New project 2\ai-cost-optimization-middleware\frontend"
copy .env.example .env.local
# edit VITE_API_URL if needed
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

## Why This Split Exists

The backend uses heavy ML dependencies:

- `llmlingua`
- `sentence-transformers`
- `transformers`
- `torch`
- `scipy`
- `scikit-learn`

These do not belong in the browser/frontend deployment. Render handles the Python optimizer container; Vercel serves only static dashboard assets.

## Safety Notes

- Do not commit API keys.
- Do not commit HuggingFace caches or model weights.
- Do not deploy local SQLite files.
- Keep `CORS_ORIGINS` scoped to your Vercel URL once deployed.
