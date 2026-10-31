import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
modelling_table = pd.read_csv("data/modelling_table.csv")
hub_profile = df.groupby("hub_id", as_index=False).agg(
  shipments = ("hub_id", "count"),
  mean_delivery_hours = ("delivery_hours", "mean"),
  delay_rate = ("is_delayed", "mean"),
  mean_distance_km = ("distance_km", "mean"),
  mean_n_stops = ("n_stops", "mean")
)
hub_profile = hub_profile.sort_values(by="hub_id", ascending=True)
hub_profile.to_csv("outputs/hub_profile.csv", index=False)
feature_cols = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
hub_profile = hub_profile.sort_values(by="hub_id", ascending=True).reset_index(drop=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(hub_profile[feature_cols])
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
hub_profile["cluster"] = kmeans.fit_predict(X_scaled)
dist_matrix = squareform(pdist(X_scaled, metric="euclidean"))
results = []
for k in [2, 3, 4, 5]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    same_dists, diff_dists = [], []
    for i in range(24):
      for j in range(i + 1, 24):
        dist = np.linalg.norm(X_scaled[i] - X_scaled[j])
        if labels[i] == labels[j]:
            same_dists.append(dist)
        else:
            diff_dists.append(dist)
    across_over_within = np.mean(diff_dists) / np.mean(same_dists)
    counts = pd.Series(labels).value_counts().sort_values(ascending=False)
    cluster_sizes_str = "-".join(counts.astype(str))
    results.append({
      "K": k, 
      "across_over_within": ratio, 
      "cluster_sizes": cluster_sizes_str
    })
separation_df = pd.DataFrame(results)
separation_df.to_csv("outputs/part_d_separation.csv", index=False)
ordered_clusters = (hub_profile.groupby("cluster")["delay_rate"].mean().sort_values(ascending=True).index)
delay_rate_names = ["Standard", "Watch", "Escalate"]
cluster_name_map = dict(zip(ranked_clusters, delay_rate_names))
hub_profile["cluster_segment"] = hub_profile["cluster"].map(cluster_name_map)
hub_meta = (df[["hub_id", "hub_city", "hub_tier"]].drop_duplicates().sort_values("hub_id"))
hub_segments = pd.merge(hub_profile[["hub_id", "cluster_segment"]], hub_meta, on="hub_id")
hub_segments = hub_segments[["hub_id", "segment", "hub_city", "hub_tier"]].sort_values("hub_id")
hub_segments.to_csv("outputs/hub_segments.csv", index=False)
segment_profile = hub_profile.groupby("segment", as_index=False).agg(
  hubs = ("hub_id", "count"),
  mean_delivery_hours = ("delivery_hours", "mean"),
  delay_rate = ("is_delayed", "mean"),
  mean_distance_km = ("distance_km", "mean"),
  mean_n_stops = ("n_stops", "mean")
)
tier_to_implied_segment = {
    "Tier 1": "Standard",
    "Tier 2": "Watch",
    "Tier 3": "Escalate",
    1: "Standard",
    2: "Watch",
    3: "Escalate",
}
mismatch_mask = (hub_segments["segment"] != hub_segments["hub_tier"].map(tier_to_implied_segment))
tier_vs_segment_mismatch = hub_segments[mismatch_mask].copy()
tier_vs_segment_mismatch.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)
final_config = pd.read_csv("outputs/part_c_final_config.csv")
