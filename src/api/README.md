# GridLock R2 — API & Frontend Architecture

This document describes the end-to-end architecture of the GridLock R2 web interface, encompassing the FastAPI backend (`src/api`) and the static HTML/JS frontend (`docs/index.html`).

## Architecture Overview

**Backend (FastAPI)**:
- Acts as a serving layer that lazily loads the winning Machine Learning model from `checkpoints/`.
- Computes predictions and aggregates spatial data on the fly based on HTTP requests.
- Designed with modular routers: `System`, `Predictions`, and `Analytics`.

**Frontend (Vanilla JS + Leaflet + Chart.js)**:
- A fully static, lightweight HTML dashboard.
- Uses native `fetch` API to asynchronously poll the backend and update the DOM dynamically.
- Eliminates the need for a heavy frontend framework like React/Next.js, ensuring maximum speed and simplicity for the hackathon demo.

## API Routes & Frontend Consumption

**Yes, every route we built is now fully consumed by the frontend!** Here is how the UI utilizes each endpoint:

### 1. System Routes (`/api/v1/system/*`)
*Used to check system health and manage raw data.*

- **`GET /health`**
  - **Returns**: Basic `{"status": "ok"}` ping.
  - **Frontend Usage**: Useful for deployment/load-balancer health checks.
- **`GET /model-status`**
  - **Returns**: Metadata about the currently loaded ML checkpoint (e.g., LightGBM), time resolution, and feature list.
  - **Frontend Usage**: Debugging, and can be used to dynamically display model versioning in the UI footer.
- **`POST /upload`**
  - **Returns**: Success message.
  - **Frontend Usage**: Used by judges or administrators via curl/postman to upload and overwrite `data/raw/*.csv` files before triggering a new pipeline run.

### 2. Prediction Routes (`/api/v1/predictions/*`)
*Core inference engine powering the spatial map.*

- **`GET /hotspots?date=YYYY-MM-DD&hour=H&top_k=N`**
  - **Returns**: A ranked array of the highest-risk enforcement zones for a specific hour, enriched with latitude, longitude, and NLP dispatch strategies.
  - **Frontend Usage**: Powers the **Interactive Leaflet Map** (rendering red/orange/green circles based on risk) and the **"Top 10 Enforcement Zones" table**.

### 3. Analytics Routes (`/api/v1/analytics/*`)
*Provides aggregated KPIs and drill-down insights.*

- **`GET /metrics?date=YYYY-MM-DD&hour=H`**
  - **Returns**: City-wide totals for the given hour (Total Violations, Critical Junctions, Active Hotspots).
  - **Frontend Usage**: Populates the **3 big KPI Scorecards** at the top of the dashboard.
- **`GET /daily-trend?date=YYYY-MM-DD`**
  - **Returns**: An array of 24 floats representing the total predicted violations across the entire city for every hour of the selected date.
  - **Frontend Usage**: Renders the large **"24-Hour Violation Trend"** line chart.
- **`GET /feature-importance`**
  - **Returns**: The global mean absolute SHAP values from the currently trained model's report.
  - **Frontend Usage**: Renders the **"Global Feature Importance (SHAP)"** horizontal bar chart.
- **`GET /zones/{zone_id}/trend`**
  - **Returns**: A 7-day historical trend of actual violations for a specific zone ID.
  - **Frontend Usage**: Consumed by the **🔍 button** inside the Hotspots table. When clicked, it opens a targeted, interactive line chart modal for that specific zone.

## How to Run

1. **Start the Backend**:
   ```bash
   uvicorn src.api.main:app --port 9000 --reload
   ```
   *The API will be available at `http://127.0.0.1:9000`. Auto-generated OpenAPI docs are at `http://127.0.0.1:9000/docs`.*

2. **Serve the Frontend**:
   ```bash
   python -m http.server 9090 --directory docs
   ```
   *The interactive dashboard will be available at `http://localhost:9090`.*
