"""Part D (3/3) - score new parcels and turn the probabilities into an action list."""
import joblib
import pandas as pd
from src.shipment_prep import clean_shipments, build_features

new = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")
model = joblib.load("models/delay_model.pkl")
threshold = float(pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0])
feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
medians = pd.read_csv("models/impute_values.csv").set_index("column")["median"].astype(float).to_dict()
segments = pd.read_csv("outputs/hub_segments.csv")[["hub_id", "segment"]]

clean, _ = clean_shipments(new, medians=medians)           # same cleaning as training, stored medians
feats = build_features(clean, hubs)
out = feats[["shipment_id", "hub_id"]].merge(segments, on="hub_id", how="left")
out["delay_probability"] = model.predict_proba(feats[feature_order])[:, 1].round(4)
out["flagged"] = (out["delay_probability"] >= threshold).astype(int)
out["action"] = "No action"
out.loc[out["flagged"] == 1, "action"] = "Proactive customer call"
out.loc[(out["flagged"] == 1) & (out["segment"] == "Escalate"), "action"] = "Escalate to hub manager"
out = out[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]]
out.to_csv("outputs/new_shipment_actions.csv", index=False)
print(out["action"].value_counts().to_string())
