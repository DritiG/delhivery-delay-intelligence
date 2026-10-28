"""Part D (1/3) - hub profile, K-Means segments, separation evidence, tier check."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

table = pd.read_csv("data/modelling_table.csv")
hubs = pd.read_csv("data/hubs.csv")

# 1. hub profile (groupby sorts by hub_id ascending)
profile = table.groupby("hub_id").agg(
    shipments=("shipment_id", "count"), mean_delivery_hours=("delivery_hours", "mean"),
    delay_rate=("is_delayed", "mean"), mean_distance_km=("distance_km", "mean"),
    mean_n_stops=("n_stops", "mean")).reset_index().sort_values("hub_id").reset_index(drop=True)
profile.round(4).to_csv("outputs/hub_profile.csv", index=False)

# 2. K-Means on the scaled four profile columns
COLS = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
Z = StandardScaler().fit_transform(profile[COLS])
km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(Z)
profile["cluster"] = km.labels_

# 3. separation evidence, K = 2..5
def across_over_within(Z, labels):
    d = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))
    iu = np.triu_indices(len(Z), k=1)
    same = labels[:, None] == labels[None, :]
    return d[iu][~same[iu]].mean() / d[iu][same[iu]].mean()

sep = []
for K in [2, 3, 4, 5]:
    lab = KMeans(n_clusters=K, random_state=42, n_init=10).fit(Z).labels_
    sizes = sorted(np.bincount(lab), reverse=True)
    sep.append(dict(K=K, across_over_within=across_over_within(Z, lab), cluster_sizes="-".join(map(str, sizes))))
pd.DataFrame(sep).round(4).to_csv("outputs/part_d_separation.csv", index=False)

# 4. segment names by mean delay rate (lowest = Standard, middle = Watch, highest = Escalate)
order = profile.groupby("cluster")["delay_rate"].mean().sort_values().index
names = dict(zip(order, ["Standard", "Watch", "Escalate"]))
profile["segment"] = profile["cluster"].map(names)
seg = profile[["hub_id", "segment"]].merge(hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left")
seg.to_csv("outputs/hub_segments.csv", index=False)

sp = profile.groupby("segment").agg(hubs=("hub_id", "count"), mean_delivery_hours=("mean_delivery_hours", "mean"),
                                    delay_rate=("delay_rate", "mean"), mean_distance_km=("mean_distance_km", "mean"),
                                    mean_n_stops=("mean_n_stops", "mean")).reindex(["Standard", "Watch", "Escalate"])
sp.reset_index().round(4).to_csv("outputs/segment_profile.csv", index=False)

implied = {1: "Standard", 2: "Watch", 3: "Escalate"}
seg[seg["segment"] != seg["hub_tier"].map(implied)].to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)

for f in ["hub_profile", "part_d_separation", "hub_segments", "segment_profile", "tier_vs_segment_mismatch"]:
    print(f); print(pd.read_csv(f"outputs/{f}.csv").to_string(index=False))
