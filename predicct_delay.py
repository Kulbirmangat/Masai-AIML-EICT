import joblib
import numpy as np
import pandas as pd
from src.shipment_prep import clean_shipments, build_features
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
cleaned_parcels, _ = clean_shipments(new_shipments, medians=stored_medians)
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
