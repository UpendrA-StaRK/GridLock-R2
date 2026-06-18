"""
src/models/clustering.py
GridLock R2 — PS1: Parking-Induced Congestion

Geospatial clustering pipeline:
  1. DBSCAN on (latitude, longitude) → zone_id per row
  2. KDE density surface over all violations
  3. Congestion Impact Score (CIS) per zone — formula from configs/eval.yaml

Usage sequence:
  a. Run notebooks/02_cluster_tuning.ipynb first to pick eps/min_samples via grid search
  b. Commit the chosen params to configs/model.yaml (dbscan section)
  c. Then call run_clustering() from notebooks/03_clustering.ipynb (to be created)

Rules (from claude.md):
  - NEVER commit DBSCAN eps without running 02_cluster_tuning.ipynb first
  - NEVER change the CIS formula without versioning it in configs/eval.yaml
  - DBSCAN noise points (label=-1) → kept as 'sparse zone', CIS weight = 0.5
  - All params read from configs/ — never hardcoded here
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# ── Config loaders ────────────────────────────────────────────────────────────

def load_eval_config(config_path: str | Path = "configs/eval.yaml") -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"configs/eval.yaml not found at '{path.resolve()}'")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_model_config(config_path: str | Path = "configs/model.yaml") -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"configs/model.yaml not found at '{path.resolve()}'")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ── DBSCAN grid search (for 02_cluster_tuning.ipynb) ─────────────────────────

def dbscan_grid_search(
    df: pd.DataFrame,
    eps_values: list[float],
    min_samples_values: list[int],
    sample_size: int = 50_000,
    random_state: int = 42,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> pd.DataFrame:
    """
    Grid search over DBSCAN eps × min_samples to find the best clustering params.

    Uses a stratified sample (default 50k rows) for speed — full 268k takes too long
    for interactive grid search. Silhouette score computed on non-noise points only.

    Args:
        df: Feature-engineered DataFrame with lat/lon columns.
        eps_values: List of epsilon values to try (in degrees; 0.005° ≈ 500m).
        min_samples_values: List of min_samples values to try.
        sample_size: Number of rows to sample for grid search.
        random_state: Random seed for reproducible sampling.
        lat_col: Name of latitude column.
        lon_col: Name of longitude column.

    Returns:
        results_df: DataFrame with columns:
            eps, min_samples, n_clusters, n_noise, noise_pct,
            silhouette_score, avg_cluster_size
    """
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(df), size=min(sample_size, len(df)), replace=False)
    sample = df.iloc[idx][[lat_col, lon_col]].copy()

    # Haversine-approximated: scale coords so 1 unit ≈ 1 km
    # Simple Euclidean on scaled coords is sufficient for DBSCAN in a small city bbox
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(sample[[lat_col, lon_col]].values)

    results: list[dict[str, Any]] = []
    total = len(eps_values) * len(min_samples_values)

    logger.info(
        f"DBSCAN grid search: {len(eps_values)} eps × {len(min_samples_values)} min_samples "
        f"= {total} combinations on {len(sample):,}-row sample"
    )

    with tqdm(total=total, desc="DBSCAN grid search", unit="combo") as pbar:
        for eps in eps_values:
            for min_samples in min_samples_values:
                db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
                labels = db.fit_predict(coords_scaled)

                n_clusters  = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise     = int((labels == -1).sum())
                noise_pct   = round(n_noise / len(labels) * 100, 1)

                # Silhouette only meaningful if ≥2 clusters and not all noise
                non_noise_mask = labels != -1
                sil = float("nan")
                if n_clusters >= 2 and non_noise_mask.sum() >= 2:
                    try:
                        sil = silhouette_score(
                            coords_scaled[non_noise_mask],
                            labels[non_noise_mask],
                            sample_size=min(10_000, non_noise_mask.sum()),
                            random_state=random_state,
                        )
                        sil = round(float(sil), 4)
                    except Exception:
                        sil = float("nan")

                avg_cluster_size = (
                    round((len(labels) - n_noise) / n_clusters, 1)
                    if n_clusters > 0
                    else float("nan")
                )

                results.append({
                    "eps":              eps,
                    "min_samples":      min_samples,
                    "n_clusters":       n_clusters,
                    "n_noise":          n_noise,
                    "noise_pct":        noise_pct,
                    "silhouette_score": sil,
                    "avg_cluster_size": avg_cluster_size,
                })
                pbar.update(1)

    results_df = pd.DataFrame(results).sort_values(
        "silhouette_score", ascending=False
    ).reset_index(drop=True)

    logger.info(
        f"Grid search complete. Best: eps={results_df.iloc[0]['eps']}, "
        f"min_samples={results_df.iloc[0]['min_samples']}, "
        f"silhouette={results_df.iloc[0]['silhouette_score']:.4f}, "
        f"n_clusters={results_df.iloc[0]['n_clusters']}"
    )
    return results_df


# ── Final DBSCAN run (called after params committed) ──────────────────────────

def run_clustering(
    df: pd.DataFrame,
    eps: float,
    min_samples: int,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Run DBSCAN with committed params on the full dataset.

    Assigns zone_id to every row. Noise points (DBSCAN label=-1) are kept
    as 'sparse zone' per eval.yaml noise_zones policy.

    Args:
        df: Feature-engineered DataFrame (output of extract_row_features).
        eps: Committed epsilon value (from 02_cluster_tuning.ipynb).
        min_samples: Committed min_samples (from 02_cluster_tuning.ipynb).
        lat_col: Latitude column name.
        lon_col: Longitude column name.
        random_state: For reproducibility logging.

    Returns:
        df_zoned: DataFrame with zone_id column added.
        stats: Dict with cluster stats.
    """
    logger.info(
        f"Running DBSCAN on full dataset ({len(df):,} rows) | "
        f"eps={eps}, min_samples={min_samples}"
    )

    df = df.copy()

    with tqdm(total=3, desc="DBSCAN clustering", unit="step", leave=True) as pbar:

        pbar.set_description("Scaling coordinates")
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(df[[lat_col, lon_col]].values)
        pbar.update(1)

        pbar.set_description("Fitting DBSCAN")
        db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = db.fit_predict(coords_scaled)
        pbar.update(1)

        pbar.set_description("Assigning zone_id")
        df["zone_id"] = labels.astype("int32")
        pbar.update(1)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int((labels == -1).sum())
    noise_pct  = round(n_noise / len(labels) * 100, 2)

    # Silhouette on full dataset (sampled for speed)
    non_noise_mask = labels != -1
    sil = float("nan")
    if n_clusters >= 2 and non_noise_mask.sum() >= 2:
        try:
            sil = silhouette_score(
                coords_scaled[non_noise_mask],
                labels[non_noise_mask],
                sample_size=min(20_000, non_noise_mask.sum()),
                random_state=random_state,
            )
            sil = round(float(sil), 4)
        except Exception as e:
            logger.warning(f"Silhouette score failed: {e}")

    stats: dict[str, Any] = {
        "eps":              eps,
        "min_samples":      min_samples,
        "n_clusters":       n_clusters,
        "n_noise":          n_noise,
        "noise_pct":        noise_pct,
        "silhouette_score": sil,
        "total_rows":       len(df),
        "zone_id_min":      int(df["zone_id"].min()),
        "zone_id_max":      int(df["zone_id"].max()),
    }

    logger.info(
        f"✓ DBSCAN complete: {n_clusters} clusters | "
        f"{n_noise:,} noise rows ({noise_pct}%) → zone_id=-1 (sparse zone) | "
        f"silhouette={sil}"
    )
    return df, stats


# ── KDE density surface ───────────────────────────────────────────────────────

def compute_kde(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    grid_points: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a 2D KDE density surface over violation locations.

    Used for heatmap visualisation (dashboard + static output).
    Not used as an ML feature — for display only.

    Args:
        df: DataFrame with lat/lon columns.
        grid_points: Number of grid points per axis for the density surface.

    Returns:
        lat_grid: 1D array of latitude grid points.
        lon_grid: 1D array of longitude grid points.
        density:  2D array of density values [grid_points × grid_points].
    """
    logger.info(f"Computing KDE on {len(df):,} points ({grid_points}×{grid_points} grid) ...")

    lats = df[lat_col].values
    lons = df[lon_col].values

    lat_grid = np.linspace(lats.min(), lats.max(), grid_points)
    lon_grid = np.linspace(lons.min(), lons.max(), grid_points)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

    positions = np.vstack([lat_mesh.ravel(), lon_mesh.ravel()])
    kernel    = gaussian_kde(np.vstack([lats, lons]))

    with tqdm(total=1, desc="KDE evaluation", unit="surface", leave=True) as pbar:
        density = kernel(positions).reshape(grid_points, grid_points)
        pbar.update(1)

    logger.info(f"✓ KDE complete: density range [{density.min():.4f}, {density.max():.4f}]")
    return lat_grid, lon_grid, density


# ── CIS computation ───────────────────────────────────────────────────────────

def compute_cis(
    df: pd.DataFrame,
    eval_config: dict[str, Any],
) -> pd.DataFrame:
    """
    Compute the Congestion Impact Score (CIS) per zone.

    Formula v1.0 (from configs/eval.yaml):
        CIS(zone) = violation_density_norm(zone) × junction_weight(zone)

    Where:
        violation_density_norm = zone_violation_count / max_zone_violation_count
        junction_weight = 1.5 if any violation in zone has is_at_junction=1, else 1.0

    Noise zone (zone_id = -1):
        Gets cis_weight_override = 0.5 (from eval.yaml noise_zones section)

    Args:
        df: DataFrame with zone_id and is_at_junction columns (post-DBSCAN).
        eval_config: Parsed configs/eval.yaml dict.

    Returns:
        cis_df: DataFrame with columns [zone_id, violation_count,
                violation_density_norm, junction_weight, cis_score,
                formula_version, priority_tier]
    """
    cis_cfg   = eval_config.get("cis", {})
    noise_cfg = eval_config.get("noise_zones", {})
    formula_version   = cis_cfg.get("formula_version", "1.0")
    noise_cis_weight  = float(noise_cfg.get("cis_weight_override", 0.5))

    logger.info(
        f"Computing CIS v{formula_version} for {df['zone_id'].nunique()} zones ..."
    )

    with tqdm(total=4, desc="Computing CIS", unit="step", leave=True) as pbar:

        # Step 1: Violation count per zone
        pbar.set_description("Counting violations per zone")
        zone_counts = (
            df.groupby("zone_id", observed=True)
            .size()
            .reset_index(name="violation_count")
        )
        pbar.update(1)

        # Step 2: violation_density_norm = count / max_count
        pbar.set_description("Normalising density")
        max_count = zone_counts["violation_count"].max()
        zone_counts["violation_density_norm"] = (
            zone_counts["violation_count"] / max_count
        ).round(6)
        pbar.update(1)

        # Step 3: junction_weight per zone
        pbar.set_description("Computing junction weights")
        zone_junction = (
            df.groupby("zone_id", observed=True)["is_at_junction"]
            .max()  # 1 if ANY violation in zone is at a junction
            .reset_index(name="has_junction")
        )
        zone_counts = zone_counts.merge(zone_junction, on="zone_id", how="left")
        zone_counts["junction_weight"] = np.where(
            zone_counts["has_junction"] == 1, 1.5, 1.0
        )
        pbar.update(1)

        # Step 4: CIS = density_norm × junction_weight
        # Noise zone override: multiply by noise_cis_weight instead
        pbar.set_description("Computing CIS scores")
        zone_counts["cis_score"] = (
            zone_counts["violation_density_norm"] * zone_counts["junction_weight"]
        )
        # Apply noise zone CIS override
        noise_mask = zone_counts["zone_id"] == -1
        if noise_mask.any():
            # CIS for noise zone = density_norm * noise_cis_weight (not junction_weight)
            zone_counts.loc[noise_mask, "junction_weight"] = noise_cis_weight
            zone_counts.loc[noise_mask, "cis_score"] = (
                zone_counts.loc[noise_mask, "violation_density_norm"] * noise_cis_weight
            )
            logger.info(
                f"  Noise zone (zone_id=-1): cis_weight_override={noise_cis_weight} applied"
            )
        pbar.update(1)

    zone_counts["formula_version"] = formula_version

    # Priority tier (for display — not for model training)
    max_cis = zone_counts["cis_score"].max()
    zone_counts["priority_tier"] = pd.cut(
        zone_counts["cis_score"],
        bins=[-0.001, 0.4 * max_cis, 0.7 * max_cis, max_cis + 0.001],
        labels=["LOW", "MEDIUM", "HIGH"],
    )

    # Phase 3 addition: cis_score_norm — 0-1 normalized CIS score.
    # Additive column only. cis_score (raw) is preserved unchanged.
    # cis_score_norm = cis_score / max(cis_score) across all zones.
    # Useful for interpretable display: "Zone X has CIS of 0.82 (82% of max impact)".
    # Rankings are identical to raw cis_score — normalization is a monotone transform.
    zone_counts["cis_score_norm"] = (
        (zone_counts["cis_score"] / max_cis).clip(0.0, 1.0).round(6)
        if max_cis > 0
        else 0.0
    )

    logger.info(
        f"✓ CIS complete: "
        f"HIGH={( zone_counts['priority_tier'] == 'HIGH').sum()} zones | "
        f"MEDIUM={(zone_counts['priority_tier'] == 'MEDIUM').sum()} zones | "
        f"LOW={(  zone_counts['priority_tier'] == 'LOW').sum()} zones | "
        f"cis_score_norm range [{zone_counts['cis_score_norm'].min():.4f}, "
        f"{zone_counts['cis_score_norm'].max():.4f}]"
    )
    return zone_counts


# ── Save / load helpers ───────────────────────────────────────────────────────

def save_cluster_stats(
    stats: dict[str, Any],
    output_path: str | Path = "data/processed/cluster_stats.json",
) -> None:
    """Save DBSCAN run stats to JSON."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info(f"Cluster stats saved → '{out}'")


def save_cis_table(
    cis_df: pd.DataFrame,
    output_path: str | Path = "data/processed/cis_table.parquet",
) -> None:
    """Save the CIS table as parquet."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cis_df.to_parquet(out, index=False)
    logger.info(f"CIS table saved → '{out}'")
