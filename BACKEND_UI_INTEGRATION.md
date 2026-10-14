# Connecting the ML package to a backend and UI

## Which file do I use?

**Use `api.py` as the backend API bridge for the UI.** It loads the saved models from `models/`, reads the bundled demonstration data from `data/`, calls the ML/inventory logic in `src/phc_ml.py`, and returns JSON responses. Do not load `.joblib` model files directly in React/browser code.

- `api.py`: HTTP API endpoints that a UI or existing backend calls.
- `src/phc_ml.py`: underlying Python demand forecasting and inventory-planning functions.
- `models/medicine_demand_model.joblib`: trained demand-regression pipeline.
- `models/stockout_next_day_classifier.joblib`: trained next-day stock-out classifier.
- `frontend/PHCForecastExample.jsx`: minimal React example calling the API.

## Run the API on Windows

Open a terminal at the extracted `PHC_Medicine_ML_Final_Package` folder.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

If a compatible virtual environment already exists, activate it and install `requirements.txt` rather than creating another one.

Verify these URLs in a browser:

- `http://127.0.0.1:8000/health` — API and model health
- `http://127.0.0.1:8000/catalog` — PHCs and medicines available in the demo data
- `http://127.0.0.1:8000/docs` — interactive API documentation (Swagger UI)

Keep this terminal running while the UI uses the API.

## Endpoints

### `GET /health`
Returns service/model status, the last history date and a synthetic-data warning.

### `GET /catalog`
Returns PHC IDs, PHC/medicine combinations, medicine names and units.

### `GET /medicines?phc_id=PHC-DEMO-001`
Returns medicine options; optional `phc_id` filter.

### `POST /forecast`
Example JSON request:

```json
{
  "phc_id": "PHC-DEMO-001",
  "medicine_id": "MED001",
  "horizon_days": 14,
  "crisis": false,
  "current_stock": 100
}
```

Returns a forecast, daily inventory projection, expiry-risk events and reorder/stock-out summary. `current_stock` is optional. `crisis` is an explicit user-declared scenario applied to forecast days; it is not automatic outbreak detection.

### `POST /stockout-risk`
Example JSON request:

```json
{
  "phc_id": "PHC-DEMO-001",
  "medicine_id": "MED001",
  "crisis": false
}
```

Returns the next-day classifier score, selected threshold and flag. The score is **not probability-calibrated** and is trained/tested only on synthetic labels.

## Connect a React/Vite UI

The example component is `frontend/PHCForecastExample.jsx`. Copy it into the frontend app and render `<PHCForecastExample />`. Example `.env` setting:

```text
VITE_PHC_API_URL=http://127.0.0.1:8000
```

Restart the Vite development server after changing `.env`. The example uses browser `fetch()` to call `POST /forecast` and render the summary and daily projection.

If your existing backend is Node.js/Express, keep Python as a separate ML service and have Express call `http://127.0.0.1:8000/forecast` or `/stockout-risk`, then expose your own frontend-facing route. Avoid duplicating the Python model logic in JavaScript. In production, run services behind a proper server/network boundary, configure allowed CORS origins, add authentication/authorization and input logging, and do not expose the development server directly to the public internet.

## Important limitation

The included facility-level history, demand, stock, deliveries, expiry and labels are **synthetic demonstration data**. The API is suitable for local UI integration and project demonstration only; it is not validated for live medicine procurement or clinical decisions. Replace the bundled data with authorized real operational records, review the assumptions and retrain/revalidate before any real-world use.


## If your existing project backend is Node.js/Express

Use the optional example `integrations/express_ml_proxy.js`. It creates routes that proxy requests to the Python API, so the architecture is:

`React UI -> existing Express backend (/api/ml/...) -> Python FastAPI (127.0.0.1:8000) -> saved model`

In your existing `server.js`, after creating `app` and enabling JSON parsing, mount the router:

```js
app.use(express.json());
app.use('/api/ml', require('./integrations/express_ml_proxy'));
```

Copy `integrations/express_ml_proxy.js` into the corresponding location in your backend (adjust the relative `require` path if needed). Set `PHC_ML_API_URL` for the Node server if FastAPI runs at a different internal address. The snippet expects Node.js 18+ because it uses built-in `fetch`.

If this is only a local demo and you don't need Express in the middle, the included React example can call FastAPI directly. For deployment, prefer the Express proxy or another authenticated backend, restrict network exposure and add authorization/validation appropriate to your application.
