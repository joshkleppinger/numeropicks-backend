# NumeroPicks API — Backend

FastAPI backend for numeropicks.com. Serves prediction, scraping, and accuracy data.

## Local development

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Open http://localhost:8000/docs for the interactive API explorer.

## Deploy to Render (free tier)

1. Push this folder to a GitHub repo (e.g. `numeropicks-backend`)
2. Go to render.com → New → Web Service
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` — just click **Deploy**
5. Your API will be live at `https://numeropicks-api.onrender.com`

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/games` | All game metadata + row counts |
| GET | `/history/{game_key}?limit=20` | Recent draws |
| POST | `/scrape/{game_key}` | Fetch latest results (background) |
| POST | `/scrape-all` | Scrape all 3 games (background) |
| GET | `/scrape-status` | Last scrape time + staleness |
| GET | `/predict/{game_key}` | Run analysis, return 5 tickets |
| GET | `/accuracy/{game_key}` | Accuracy stats + recent results |
| GET | `/next-draw/{game_key}` | Next draw date (friendly string) |

## Game keys

- `powerball`
- `megamillions`
- `superlotto`

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NUMERO_DATA_DIR` | `/data/numero` | Where CSVs are stored |
| `PORT` | `8000` | Set automatically by Render |
