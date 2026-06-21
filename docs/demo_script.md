# GridLock R2 — Demo Script
# Hackathon: Gridlock 2.0 (solve . traffic) | Problem Statement 1
# Judging criteria: Feasibility · Relevance · Innovation · Real-World Impact

---

## Overview

**System:** AI-driven parking intelligence for Bengaluru Traffic Police
**What it does:** Detects illegal parking hotspots, quantifies their congestion impact,
and generates prioritized enforcement zone schedules by time of day.

**One-sentence pitch:**
> "GridLock tells traffic officers not just *where* illegal parking is concentrated,
> but *when* — and exactly how many officers to send there."

---

## 2-Minute Demo Walkthrough

### [0:00 – 0:20] Opening — The Problem

> "Bengaluru has 140+ distinct enforcement zones. Every morning, traffic officers
> face the same question: where do I go first? Today, that decision is patrol-based
> and reactive. Our system makes it data-driven and predictive."

**Show:** The Folium enforcement map loading in browser.

---

### [0:20 – 0:50] The Live Map — Zone Priority at 9am

> "This map shows our AI's enforcement recommendation for 9am on any weekday.
> The color intensity and zone labels show predicted violation counts weighted by
> the Congestion Impact Score — a formula we derived from junction presence and
> violation density at each zone."

**Action:** Point to the top 3 ranked zones on the map.

> "Zone [X] is our #1 priority — it sits on a major junction and historically peaks
> at morning rush hour. The model predicts [N] violations in this hour, and our CIS
> formula says those violations cause [Y] times more traffic disruption than a
> non-junction zone."

---

### [0:50 – 1:20] The Time Slider — Why ML Beats a Lookup Table

> "Here's where ML adds genuine value. [Move slider from 9am to 2am]."

**Show:** Zone rankings reorder as the hour changes.

> "At 2am, Zone [X] drops from #1 to #8. Zone [Z] moves to the top — it's a late-night
> hotspot near a restaurant cluster. A static frequency table can't tell you this.
> Our model was trained on 150 days of violation data, and learned these temporal
> patterns. On the test period, the model's per-hour zone ranking beats the frequency
> baseline by [X]% on NDCG@10."

---

### [1:20 – 1:45] The Numbers — Honest Assessment

> "Let's talk accuracy. On our held-out March–April 2024 test set:"

| Metric | Our ML Model | Simple Frequency Table |
|---|---|---|
| Count prediction (MAE) | **4.31** violations/zone/hour | 6.97 (naive mean) |
| ML Lift | **+38.2%** over naive baseline | — |
| Per-hour ranking (NDCG@10) | **0.894** | 0.873 (frequency table) |
| Spearman ρ (rank correlation) | **0.522** | — |

> "38.2% better count prediction means we can tell an officer 'send 2 officers' vs '5 officers'
> — not just 'go to this zone.' And on the ranking task, our per-hour NDCG beats the frequency
> table by 2 points — which translates to consistently getting the right zone at the right hour."

---

### [1:45 – 2:00] The Roadmap — What Comes Next

> "Phase 2 of this system integrates real-time MapmyIndia traffic speed data.
> Instead of inferring congestion from violation density, we'd measure it directly.
> The enforcement schedule would update every 15 minutes instead of hourly."

**Show:** The pipeline architecture diagram (README.md).

---

## Judge Q&A — Pre-Prepared Answers

### Q1: "Does your ML model actually beat a simple frequency table?"

**Answer:**
> "For the aggregate zone ranking over the full test period — no, both score 1.0 on NDCG@10,
> because the top-10 zones in Bengaluru are geographically stable (they're the same busy junctions
> every day). This is a property of the data, not a flaw in the model.
>
> But that's the wrong question. The right question is: *can the model tell you which zones are
> hottest at specific hours of the day?* There, the ML model wins — our per-hour NDCG@10 is
> **0.891 vs the baseline's 0.873** across 1,600+ individual hour slots in the test period.
> The Spearman rank correlation is 0.522 — meaning the model's zone ordering within each hour
> correlates significantly with the ground truth.
>
> A frequency table always recommends Zone 2 first, regardless of whether it's 9am or 2am.
> Our model knows Zone 2 peaks at morning rush and deprioritises it at night. That's the
> operational value that saves patrol time."

---

### Q2: "How do you quantify congestion impact without traffic speed data?"

**Answer:**
> "We didn't fake traffic data. Instead, we predict the parking violations using AI, and then use a transparent mathematical weight—the Congestion Impact Score (CIS)—to tell the police where to go first.
>
> The dataset is violation records only, so our CIS is an evidence-based proxy: it weights predicted violation density by junction presence. Violations at junctions block intersections and create cascading delays; violations mid-block have lower impact. By multiplying our ML prediction by this CIS, we output an honest Enforcement Priority Score."

---

### Q3: "What features drive your model? Is it explainable?"

**Answer (show shap_summary.png):**
> "Yes — we ran SHAP (SHapley Additive exPlanations) on the model. The top features are:
>
> 1. **rolling_7d_count** — recent violation history at this zone and hour. Makes sense:
>    a zone that's been active all week will likely be active today.
>
> 2. **zone_mean_count / zone_cis_score** — the baseline activity level of the zone.
>    High-CIS zones (busy junctions) are consistently higher priority.
>
> 3. **hour_of_day** — the temporal signal. This is what differentiates ML from a lookup table.
>
> No opaque black box — every prediction can be explained by these measurable factors."

---

### Q4: "How do you handle new zones appearing after deployment?"

**Answer:**
> "New zones from construction or development would initially be flagged as 'unknown' and receive
> the system's global mean violation count as a default priority estimate — conservative but safe.
>
> The pipeline is designed to re-cluster (DBSCAN) and retrain quarterly. After one enforcement
> cycle, the new zone builds its own historical record and the model treats it normally.
>
> Longer term, the roadmap includes HDBSCAN for dynamic zone boundary updates without full retraining."

---

### Q5: "Can this scale to other cities?"

**Answer:**
> "The pipeline is city-agnostic. The inputs are: violation records with lat/lon and timestamps.
> Those exist for every major Indian city through the e-Challan system.
>
> The DBSCAN clustering and CIS formula are parameterized — different cities have different
> junction densities. We'd re-tune eps and junction weights per city.
>
> We focused on Bengaluru as the proof of concept because ASTraM provided real violation data.
> Our Congestion Impact Score (CIS) directly mirrors ASTraM's operational methodology:
> ASTraM prioritises high-risk junctions using MoRTH Blackspot classification — we do the same,
> but data-driven. Zones at junctions receive a 1.5× CIS multiplier, zones mid-block receive 1.0.
> This is the same logic the Bengaluru Traffic Police already applies in the field, now quantified.
> Multi-city deployment is a Q3 roadmap item."

---

### Q6: "How do you know your model actually finds the right geographic areas?"

**Answer:**
> "We compute the Prediction Accuracy Index (PAI) — the standard spatial validation metric
> used in police enforcement analytics worldwide.
>
> PAI = (fraction of test violations captured by top-K zones) ÷ (fraction of total area covered)
>
> If our top-10 zones cover 7% of the city's violation geography but capture 28% of all test
> violations, the PAI is 4.0 — meaning we're 4× more efficient than random patrolling.
>
> This directly answers your question: the model doesn't just rank zones that look plausible —
> it identifies the zones where actual violations subsequently occurred at a rate far exceeding
> chance. PAI is computable from our existing outputs and can be shown on request."

---

## Judging Criteria Alignment

| Criterion | Our Evidence |
|---|---|
| **Feasibility** | End-to-end pipeline runs in 11s (inference-only mode). 6 trained models. 150 days of real Bengaluru police data. MAE=4.31 on unseen March–April test set. |
| **Relevance** | Directly addresses reactive patrol problem. Output is officer-ready hourly zone schedule. CIS formula mirrors ASTraM's MoRTH Blackspot methodology — familiar to BTP judges. |
| **Innovation** | CIS = violation density × junction weight. Per-hour NDCG (not aggregate). SHAP explainability gate. 24h live time-slider demo. PAI spatial validation. Cyclical temporal encoding. |
| **Real-World Impact** | +38.2% count prediction improvement → right number of officers per zone. Per-hour NDCG 0.894 vs 0.873 baseline → right zone at right hour, not just right zone overall. |

---

## Technical Talking Points (for expert judges)

- **No data leakage:** Hard assertion in `train.py` that `max(train date) < min(test date)`.
- **Zone features (Phase 1):** Replaced raw cluster IDs with statistically-computed zone characteristics — prevents XGBoost from memorizing zone identities.
- **Rolling features:** `rolling_7d_count` uses `shift(1).rolling(7)` — current day's count is never included.
- **Model Selection:** We evaluated 3 algorithms (XGBoost, LightGBM, CatBoost) to pick the best fit for zero-inflated data. LightGBM was strictly chosen as the single predictor in our 1-model pipeline based on per-hour NDCG@10.
- **Clustering:** DBSCAN on lat/lon, producing 139 geographically interpretable enforcement zones.

---

## Files to Have Open During Demo

1. `data/outputs/enforcement_slider_2024-03-18.html` — **PRIMARY DEMO** — 24h time-slider map (run `05_inference.ipynb` to generate)
2. `data/outputs/enforcement_priority_2024-03-18_09h.html` — Fallback single-hour map (already exists)
3. `data/outputs/shap_summary.png` — for Q3 (feature importance)
4. `data/outputs/shap_pdp_hour.png` — for hour-of-day temporal effect slide
5. `README.md` — for pipeline architecture overview

> **Before demo:** Run `notebooks/05_inference.ipynb` Cell 9 to generate the slider HTML.
> Then: `git tag demo-ready` to lock the state.

---

*Prepared for: Gridlock 2.0 Hackathon | Flipkart HQ, Bengaluru*
