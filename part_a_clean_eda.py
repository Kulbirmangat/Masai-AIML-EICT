from src.shipment_prep import clean_shipments, build_features
clean = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")
missing_parcel_weight_before = clean['parcel_weight_kg'].count()
service_type_spellings_before = clean['service_type'].nunique()
cleaned_shipments = clean_shipments(clean)
service_type_spellings_after = cleaned_shipments['service_type'].nunique()
built_features = build_features(cleaned_shipments, hubs)
built_features["is_delayed"] = np.where(df_features["delivery_hours"] > df_features["promised_hours"], 1, 0)
built_features.to_csv("data/modelling_table.csv", index=False)
nulls_in_features_and_target = built_features[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum()
declared_value_mean_inr, declared_value_median_inr = built_features["declared_value_inr"].mean(), built_features["declared_value_inr"].median()
log_declared_value_mean, log_declared_value_median = built_features["log_declared_value"].mean(), built_features["log_declared_value"].median()
hours_q1, hours_q3 = built_features["delivery_hours"].quantile(0.25), build_features["delivery_hours"].quantile(0.75)
iqr = hours_q3 - hours_q1
hours_lower_bound, hours_upper_bound = hours_q1 - (1.5 * iqr), hours_q3 + (1.5 * iqr)
hours_outlier_count = ((built_features["delivery_hours"] > hours_upper_bound) | (built_features["delivery_hours"] < hours_lower_bound)).sum()
checks_df = pd.DataFrame({
  "missing_parcel_weight_before": missing_parcel_weight_before,
  "service_type_spellings_before": service_type_spellings_before,
  "service_type_spellings_after": service_type_spellings_after,
  "nulls_in_features_and_target": nulls_in_features_and_target,
  "declared_value_mean_inr": declared_value_mean_inr.round(4),
  "declared_value_median_inr": declared_value_median_inr.round(4),
  "log_declared_value_mean": log_declared_value_mean.round(4), 
  "log_declared_value_median": log_declared_value_median.round(4),
  "hours_q1": hours_q1.round(4),
  "hours_q3": hours_q3.round(4),
  "hours_lower_bound": hours_lower_bound.round(4),
  "hours_upper_bound": hours_upper_bound.round(4),
  "hours_outlier_count": hours_outlier_count
})]
checks_df.to_csv("outputs/part_a_checks.csv", index=False)
dimensions = ["service_type", "hub_tier", "distance_band"]
all_summaries = []
for dim in dimensions:
    summary = df_cleaned.groupby(dim).agg(
        shipments=('is_delayed', 'count'),
        delayed=('is_delayed', 'sum'),
        delay_rate=('is_delayed', 'mean'),
        mean_hours=('delivery_hours', 'mean')
    ).reset_index()
    summary = summary.rename(columns={dim: "group"})
    summary["dimension"] = dim
    summary["group"] = summary["group"].astype(str)
    summary = summary[["dimension", "group", "shipments", "delayed", "delay_rate", "mean_hours"]]
    all_summaries.append(summary)
final_summary = pd.concat(all_summaries, ignore_index=True)
final_summary.to_csv("outputs/part_a_delay_by_group.csv", index=False)
correlation_matrix = built_features.corr(numeric_only=True)
correlation_matrix.to_csv("outputs/part_a_correlations.csv", index_label="feature")
