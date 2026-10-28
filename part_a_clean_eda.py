"""Part A - data cleaning, modelling table, delay evidence."""
import os
import numpy as np
import pandas as pd
from src.shipment_prep import clean_shipments, build_features, FEATURES

os.makedirs("outputs", exist_ok=True)
raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

# --- "before" measurements (after de-duplication, before any filling) ---
dedup = raw.drop_duplicates()
before_values = pd.to_numeric(dedup["declared_value"].astype("string").str.replace("\u20b9", "", regex=False)
                              .str.replace(",", "", regex=False), errors="coerce")
sentinel_rows = int((dedup["parcel_weight_kg"] > 100).sum())
spell_before = int(dedup["service_type"].nunique())

clean, medians = clean_shipments(raw)
table = build_features(clean, hubs)
rows_after_merge = len(table)
table["is_delayed"] = (table["delivery_hours"] > table["promised_hours"]).astype(int)
table.to_csv("data/modelling_table.csv", index=False)

q1, q3 = table["delivery_hours"].quantile(0.25), table["delivery_hours"].quantile(0.75)
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
outliers = int(((table["delivery_hours"] < lo) | (table["delivery_hours"] > hi)).sum())

checks = {
    "raw_rows": len(raw),
    "duplicate_rows_removed": len(raw) - len(dedup),
    "rows_after_dedup": len(dedup),
    "missing_distance_km_before": int(dedup["distance_km"].isna().sum()),
    "missing_parcel_weight_before": int(dedup["parcel_weight_kg"].isna().sum()),
    "sentinel_weight_rows": sentinel_rows,
    "missing_declared_value_before": int(dedup["declared_value"].isna().sum()),
    "missing_payment_mode_before": int(dedup["payment_mode"].isna().sum()),
    "service_type_spellings_before": spell_before,
    "service_type_spellings_after": int(table["service_type"].nunique()),
    "declared_value_mean_inr": table["declared_value_inr"].mean(),
    "declared_value_median_inr": table["declared_value_inr"].median(),
    "log_declared_value_mean": np.log(table["declared_value_inr"]).mean(),
    "log_declared_value_median": np.log(table["declared_value_inr"]).median(),
    "median_distance_km": medians["distance_km"],
    "median_parcel_weight_kg": medians["parcel_weight_kg"],
    "median_declared_value_inr": medians["declared_value_inr"],
    "rows_after_hub_merge": rows_after_merge,
    "nulls_in_features_and_target": int(table[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum()),
    "delayed_count": int(table["is_delayed"].sum()),
    "delay_rate": table["is_delayed"].mean(),
    "hours_q1": q1, "hours_q3": q3,
    "hours_lower_bound": lo, "hours_upper_bound": hi,
    "hours_outlier_count": outliers,
}
pd.DataFrame({"metric": list(checks), "value": [round(float(v), 4) for v in checks.values()]}) \
    .to_csv("outputs/part_a_checks.csv", index=False)

# --- delay evidence ---
table["distance_band"] = pd.cut(table["distance_km"], bins=[0, 200, 400, 600, 2000],
                                labels=["0-200", "201-400", "401-600", "601+"], right=True)
parts = []
for dim in ["service_type", "hub_tier", "distance_band"]:
    g = table.groupby(dim, observed=True).agg(shipments=("is_delayed", "size"), delayed=("is_delayed", "sum"),
                                              delay_rate=("is_delayed", "mean"), mean_hours=("delivery_hours", "mean"))
    g = g.reset_index().rename(columns={dim: "group"})
    g.insert(0, "dimension", dim)
    g["group"] = g["group"].astype(str)
    parts.append(g)
by_group = pd.concat(parts, ignore_index=True).round(4)
by_group.to_csv("outputs/part_a_delay_by_group.csv", index=False)

# --- correlations ---
corr = table.drop(columns=["distance_band"]).select_dtypes(include="number").corr().round(4)
corr.to_csv("outputs/part_a_correlations.csv")
print(pd.read_csv("outputs/part_a_checks.csv").to_string(index=False))
print(by_group.to_string(index=False))
c = corr["delivery_hours"].drop("delivery_hours")
print(c.reindex(c.abs().sort_values(ascending=False).index).head(5))
print(corr.loc["is_delayed", "distance_km"], corr.loc["delivery_hours", "promised_hours"])
