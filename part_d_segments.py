import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.shipment_prep import FEATURES, clean_shipments, build_features
modelling_table = pd.read_csv("data/modelling_table.csv")
hub_profile = modelling_table.groupby("hub_id", as_index=False).agg(
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
dist_matrix = euclidean_distances(X_scaled, X_scaled)
results = []
for k in [2, 3, 4, 5]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    same_dists, diff_dists = [], []
    for i in range(len(X_scaled)):
      for j in range(i + 1, 24):
        dist = np.linalg.norm(X_scaled[i] - X_scaled[j])
        if labels[i] == labels[j]:
            same_dists.append(dist)
        else:
            diff_dists.append(dist)
    ratio = np.mean(diff_dists) / np.mean(same_dists)
    counts = pd.Series(labels).value_counts().sort_values(ascending=False)
    cluster_sizes_str = "-".join(counts.astype(str))
    results.append({
      "K": k, 
      "across_over_within": ratio, 
      "cluster_sizes": cluster_sizes_str
    })
separation_df = pd.DataFrame(results)
separation_df.to_csv("outputs/part_d_separation.csv", index=False)
ordered_clusters = (
    hub_profile.groupby("cluster")["delay_rate"].mean().sort_values(ascending=True).index)
delay_rate_names = ["Standard", "Watch", "Escalate"]
cluster_name_map = dict(zip(ordered_clusters, delay_rate_names))
hub_profile["segment"] = hub_profile["cluster"].map(cluster_name_map)
hub_meta = (modelling_table[["hub_id", "hub_city", "hub_tier"]].drop_duplicates().sort_values("hub_id"))
hub_segments = pd.merge(hub_profile[["hub_id", "segment"]], hub_meta, on="hub_id")
hub_segments = hub_segments[["hub_id", "segment", "hub_city", "hub_tier"]].sort_values("hub_id")
hub_segments.to_csv("outputs/hub_segments.csv", index=False)
segment_profile = hub_profile.groupby("segment", as_index=False).agg(
    hubs=("hub_id", "count"),
    mean_delivery_hours=("mean_delivery_hours", "mean"),
    delay_rate=("delay_rate", "mean"),
    mean_distance_km=("mean_distance_km", "mean"),
    mean_n_stops=("mean_n_stops", "mean"),
)
segment_profile.to_csv("outputs/hub_segment_profile.csv", index=False)
tier_to_implied_segment = {
    "Tier 1": "Standard",
    "Tier 2": "Watch",
    "Tier 3": "Escalate",
    1: "Standard",
    2: "Watch",
    3: "Escalate",
}
mismatch_mask = hub_segments["segment"] != hub_segments["hub_tier"].map(tier_to_implied_segment)
tier_vs_segment_mismatch = hub_segments[mismatch_mask].copy()
tier_vs_segment_mismatch.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)
config_df = pd.read_csv("outputs/part_c_final_config.csv")
row = config_df.iloc[0]
k_val = int(row["selector_k"])
max_depth_val = None if str(row["max_depth"]).lower() == "unlimited" else int(row["max_depth"])
chosen_params = {
    "n_estimators": int(row["n_estimators"]),
    "max_depth": max_depth_val,
    "min_samples_leaf": int(row["min_samples_leaf"]),
}
df = pd.read_csv("data/modelling_table.csv")
X = df[FEATURES]
y = df["is_delayed"]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
final_pipeline = Pipeline([
    ("selector", SelectKBest(score_func=f_classif, k=k_val)),
    ("model", RandomForestClassifier(random_state=42, **chosen_params))
]).fit(X_train, y_train)
joblib.dump(final_pipeline, "models/delay_model.pkl")
threshold_val = float(row["threshold"])
pd.DataFrame({"threshold": [threshold_val]}).to_csv("models/delay_threshold.csv", index=False)
pd.DataFrame({"feature": FEATURES}).to_csv("models/feature_order.csv", index=False)
impute_df = pd.DataFrame({
    "column": ["parcel_weight_kg", "distance_km", "delivery_hours"],
    "median": [
        float(modelling_table["parcel_weight_kg"].median())
        if "parcel_weight_kg" in modelling_table.columns
        else float(modelling_table["weight_kg"].median()),
        float(modelling_table["distance_km"].median()),
        float(modelling_table["delivery_hours"].median()),
    ],
})
impute_df.to_csv("models/impute_values.csv", index=False)
delay_model = joblib.load("models/delay_model.pkl")
delay_model_probs = delay_model.predict_proba(X_test)
final_pipeline_probs = final_pipeline.predict_proba(X_test)
test_rows_scored = len(X_test)
reload_max_abs_probability_diff = float(np.max(np.abs(delay_model_probs - final_pipeline_probs)))
reload_check = pd.DataFrame({
    "metric": ["test_rows_scored", "reload_max_abs_probability_diff", "threshold"],
    "value": [test_rows_scored, reload_max_abs_probability_diff, threshold_val]
})
reload_check.to_csv("outputs/part_d_reload_check.csv", index=False)
new_shipments = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")
hub_segments = pd.read_csv("outputs/hub_segments.csv")
delay_model = joblib.load("models/delay_model.pkl")
threshold_df = pd.read_csv("models/delay_threshold.csv")
feature_order_df = pd.read_csv("models/feature_order.csv")
impute_df = pd.read_csv("models/impute_values.csv")
stored_threshold = float(threshold_df["threshold"].iloc[0])
feature_order = feature_order_df["feature"].tolist()
stored_medians = dict(zip(impute_df["column"], impute_df["median"]))
cleaned_parcels = clean_shipments(new_shipments, medians=stored_medians)
features_df = build_features(cleaned_parcels, hubs)
X_new = features_df[feature_order]
delay_probability = np.round(delay_model.predict_proba(X_new)[:, 1], 4)
flagged = (delay_probability >= stored_threshold).astype(int)
output_df = pd.DataFrame({
    "shipment_id": new_shipments["shipment_id"],
    "hub_id": new_shipments["hub_id"],
    "delay_probability": delay_probability,
    "flagged": flagged,
})
output_df = output_df.merge(hub_segments[["hub_id", "segment"]], on="hub_id", how="left")
conditions = [
    (output_df["flagged"] == 1) & (output_df["segment"] == "Escalate"),
    (output_df["flagged"] == 1) & (output_df["segment"] != "Escalate"),
    (output_df["flagged"] == 0)]
choices = ["Escalate to hub manager", "Proactive customer call", "No action"]
output_df["action"] = np.select(conditions, choices, default="No action")
final_columns = ["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]
output_df = output_df[final_columns]
output_df.to_csv("outputs/new_shipment_actions.csv", index=False)
