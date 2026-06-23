# 🏁 Gridlock Hackathon R2 — Problem Statement Evaluation Report

> **Team Profile:** 1–2 people · Generalist (ML + CV + Full-Stack) · < 1 week · RTX 3050 Ti (4 GB VRAM) + i7-12700H + 16 GB RAM

---

## 🔍 Executive Summary

| | PS1: Parking Intelligence | PS2: Event Congestion | PS3: CV Violations |
|---|---|---|---|
| **Dataset size** | 109.6 MB · 298,450 rows | 4.55 MB · 8,205 rows | ❌ No dataset provided |
| **Dataset quality** | ⭐⭐⭐⭐ Rich, geo-stamped | ⭐⭐⭐ Moderate, sparse planned | ❌ N/A |
| **What you submit** | Working prototype | Working prototype | Concept note / idea only |
| **GPU needed?** | No | No | Yes (if you build it) |
| **Hardware fits?** | ✅ Fully local | ✅ Fully local | ⚠️ Partially (inference only) |
| **Feasibility (1 week)** | 🟢 HIGH | 🟡 MEDIUM | 🟢 HIGH (idea only) |
| **Finals impact** | 🟢 HIGH | 🟡 MEDIUM | 🟡 MEDIUM |
| **Overall Score** | ⭐ 1st | ⭐⭐ 2nd | ⭐⭐⭐ 3rd |

---

## 📊 Dataset Analysis

### PS1 — Police Violation Dataset (`jan to may police violation_anonymized791b166.csv`)

| Metric | Value |
|---|---|
| File size | **109.6 MB** |
| Total records | **298,450 rows** (Nov 2023 – Apr 2024, ~6 months) |
| Columns | 24 (id, lat, lon, location, vehicle type, violation type, offence code, timestamps, police station, junction, validation status…) |
| Geographic coverage | Lat: 12.80–13.29, Lon: 77.44–77.77 → **Full Bengaluru** |
| Top violations | WRONG PARKING (138,764) · NO PARKING (119,576) · PARKING IN MAIN ROAD · FOOTPATH PARKING |
| Geospatial data | ✅ Lat/Lon on every record → directly heatmappable |
| Vehicle types | Cars, bikes, heavy vehicles, maxi-cabs |
| Validation status | Present — `approved` / `NULL` labeling available |

**Quality Assessment:**
- ✅ `latitude` and `longitude` are present and valid on virtually all records
- ✅ `created_datetime` spans 6 months — enough for temporal analysis (time-of-day, day-of-week trends)
- ✅ `police_station`, `junction_name`, `center_code` — enables zone-level aggregation
- ⚠️ `description` column: 298,450/298,450 are NULL → no free-text info
- ⚠️ `closed_datetime`, `action_taken_timestamp`: entirely NULL → enforcement outcome cannot be measured
- ⚠️ No actual traffic flow data (speed, volume) → enforcement priority must be *inferred* via a Congestion Impact Score (CIS) proxy, not measured directly as true delay

**What you can build with this:**
1. Geospatial violation heatmap (Folium/Plotly/Kepler.gl)
2. Hotspot detection using DBSCAN/KDE clustering
3. Temporal analysis: peak violation hours → map to rush hours → infer congestion
4. Police station / zone-level enforcement gap scoring
5. Predictive model: which zones will have violations on a given day/time?
6. Interactive dashboard with filtering by violation type, zone, time

---

## 💻 Hardware Compatibility Analysis

**Your PC: RTX 3050 Ti · i7-12700H · 16 GB RAM · Windows**

| Task | Can Your PC Handle It? | Notes |
|---|---|---|
| Pandas/EDA on 109 MB CSV | ✅ Easily | In-memory, no GPU needed |
| Geospatial clustering (DBSCAN/KDE) | ✅ Easily | CPU-bound, fast on i7-12700H |
| scikit-learn ML models (XGBoost, RF) | ✅ Easily | All CPU/RAM, ~1 GB usage |
| Folium/Plotly interactive maps | ✅ Easily | Browser-based rendering |
| YOLOv8n/s inference (PS3) | ✅ Yes (30+ FPS) | 4 GB VRAM handles inference fine |
| YOLOv8m/l training from scratch (PS3) | ⚠️ Marginal | 4 GB VRAM is tight — batch size 4–8 only |
| Fine-tuning YOLOv8 (PS3) | ✅ Doable | With small batches + FP16 + gradient checkpointing |
| LLM inference (Llama 7B) | ⚠️ Marginal | Too large for 4 GB VRAM; CPU offload needed |

**Bottom line:** Your hardware is **fully sufficient** for PS1 and PS2 without any cloud resources. For PS3, you'd need Kaggle (free) or Colab for model training.

---

## ☁️ Cloud Resources — Should You Create New Accounts?

### Azure (New Account)
- **Free credit:** $200 USD for 30 days
- **GPU access:** ❌ Blocked by default on free tier. Core quota is ~4 vCPUs. GPU VMs need 6–24+ cores. Quota increases are not eligible on free accounts. You must upgrade to Pay-As-You-Go to unlock GPUs.
- **Verdict:** Creating a new Azure account gives you $200 for non-GPU services (storage, web apps, Azure Maps API). **GPU access is NOT guaranteed and likely blocked.** Not worth it for GPU workloads.

### GCP (New Account)
- **Free credit:** $300 USD for 90 days — the best offer
- **GPU access:** ⚠️ Requires account upgrade to paid billing (your $300 credits still apply). Need to request GPU quota separately.
- **Verdict:** Worth creating if you need GPU. After upgrading billing, a T4 GPU VM costs ~$0.35/hr. $300 = ~850 GPU-hours. For PS3 fine-tuning (a few hours), this is more than enough.

### Kaggle (Recommended — No New Account Trick Needed)
- **Free:** 30 GPU-hours/week on T4 or P100. No billing setup needed.
- **Verdict:** Best option for PS3 training. No risk of accidental charges. Use Kaggle + local RTX 3050 Ti for inference.

### Google Colab Free
- **Free:** T4 GPU, session-based (~12 hrs). Intermittent availability.
- **Verdict:** Good backup for short training runs.

> **Bottom line for new accounts:** For PS1 and PS2, you need **zero cloud resources** — your local PC is sufficient. For PS3, **Kaggle's free tier is all you need** for training. Only create a GCP account if you specifically choose PS3 and want uninterrupted, longer training sessions.

---

## 🏆 Finalist Potential Analysis

The top 10 selection is based on: **Feasibility · Relevance · Innovation · Real-World Impact**

### PS1 — Parking Intelligence
- **Why it can win:** 298K rows of rich, geo-stamped data means you can build *something that actually works* — a live interactive map with clustering, hotspot scoring, and a predictive enforcement priority model. Judges can see real Bengaluru locations on a map. That is immediately impressive and relatable to the Bengaluru Traffic Police audience.
- **Innovation angle:** Beyond just heatmaps — frame it as "Enforcement Resource Optimizer": predict which zones will have violations on a given day/time/weather and pre-position officers accordingly. Add a "Congestion Impact Score" per hotspot using proximity-to-junction weighting.
- **Risk:** Many teams may pick this (most data, clearest direction). Stand out by going beyond heatmaps to predictive enforcement + economic impact modeling.

### PS2 — Event Congestion
- **Why it can win:** The problem statement asks for *forecasting* and *recommendation* — a more sophisticated ask. If your model actually predicts impact before an event, that's powerful.
- **Why it's harder:** Only 8,205 rows. Only 467 planned events. The "recommend manpower/barricading" part has no ground truth data. You'd be building an **LLM-assisted recommendation engine** without a dataset to validate it.
- **Risk:** Small dataset = limited model confidence = judges may push back on reliability.

### PS3 — Computer Vision (Idea Only)
- **Why it can place:** No working prototype required. A beautifully structured concept note with a clear architecture (YOLOv8 + OCR pipeline + evidence generation), a working demo on a public dataset, and a polished slide deck can be very compelling.
- **Why it's riskier:** No hardware data to work with. Innovation must come from design, not results. Harder to differentiate from other idea submissions unless your concept is genuinely novel (e.g., federated learning across cameras, edge deployment on dashcams).

---

## 🎯 Final Recommendation

### 🥇 Choose Problem Statement 1 (Parking Intelligence)

**Reasoning:**

1. **Largest, richest dataset:** 298K rows with full geospatial, temporal, zone, and violation-type coverage. You have everything you need to build a polished, working prototype.

2. **No GPU required:** Everything runs on your CPU/RAM. No cloud setup delays. You can start coding Day 1.

3. **Highest demo impact:** A live Folium/Plotly map showing violation hotspots overlaid on real Bengaluru streets, with a time-of-day slider and enforcement priority scores, will visually impress judges — especially the Bengaluru Traffic Police panel who know these roads.

4. **Winnable innovation:** "Which zones should officers patrol *tomorrow morning* at 8am?" is a clear, practical, answerable question. Build a gradient boosting model (XGBoost/LightGBM) to predict violation probability by zone × time × vehicle type. This is novel yet achievable.

5. **1-week feasibility:** Day 1 = EDA + heatmap. Day 2 = Clustering. Day 3 = Predictive model. Day 4 = Dashboard. Day 5 = Refinement + video demo. Very doable for 1–2 people.

---

### Recommended Prototype Architecture for PS1

```
Data Pipeline
  └── pandas EDA → Feature Engineering (hour, weekday, zone, junction proximity)

Geospatial Layer
  └── DBSCAN hotspot clustering → Folium choropleth map (interactive)

Predictive Layer
  └── XGBoost / LightGBM: predict violation count by zone × time block

Dashboard (Streamlit or HTML)
  └── Interactive heatmap + time slider + enforcement priority score + prediction widget

Innovation Angle
  └── "Congestion Impact Score": weight violation density by road type + proximity to junction
  └── "Enforcement Gap Analysis": zones with high violations but low police presence
```

---

## ⚡ Quick Decision Matrix

If your team's priority is…

| Priority | Choose |
|---|---|
| Maximize working demo quality | **PS1** |
| Fastest path to finalist round | **PS1** |
| Lowest technical risk | **PS1** |
| Most cutting-edge AI story | **PS3** (idea) |
| Forecasting / ops research angle | **PS2** (if team has 2+ weeks) |
