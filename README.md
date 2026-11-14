# Street Watch - backend (Flask + ONNX + MySQL)

## Endpoints
| Method | Path | Purpose |
|---|---|---|
| GET | /api/health | model + database status |
| POST | /api/detect | multipart `image`, optional `conf`, `save`, `source` |
| GET | /api/history?page=1&limit=10 | saved runs |
| GET | /api/history/<id> | one run with its boxes |
| DELETE | /api/history/<id> | delete a run |
| GET | /api/stats | totals |

## Environment variables (set on Render, never in GitHub)
`DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME` - from the Aiven console.
`DB_SSL_MODE` - `REQUIRED` (default, works with Aiven) or `VERIFY` + `DB_SSL_CA` (paste ca.pem text).
Optional: `CONF_THRESHOLD`, `IOU_THRESHOLD`, `CORS_ORIGINS`, `MODEL_PATH`.

## Deploy
1. Create an Aiven free MySQL service; copy host, port, user, password, database.
2. Push this folder to GitHub (the `.env` file is git-ignored; `models/best.onnx` must be committed).
3. Render -> New Web Service -> pick the repo -> Root Directory = this folder.
   Build: `pip install -r requirements.txt`
   Start: `gunicorn app:app --workers 1 --threads 2 --timeout 120 --bind 0.0.0.0:$PORT`
   Runtime: Python 3.11 (set `PYTHON_VERSION=3.11.9`). Plan: Free.
4. Add the DB_* variables under Environment, deploy, open `/api/health`.
   Tables are created automatically on first start.

## Local run
    python -m venv venv && source venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env   # fill in
    python app.py

Free-tier notes: Render sleeps after ~15 min idle (first request takes ~50 s); Aiven free MySQL may power off when idle - the app keeps working without it and just skips saving.
