import pandas as pd
import numpy as np
PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]
def clean_shipments(raw, medians=None):
  clean_df = raw.drop_duplicates()
  clean_df["service_type"] = clean_df["service_type"].str.strip().str.title()
  clean_df["payment_mode"] = clean_df["payment_mode"].fillna("Unknown")
  clean_df["declared_value_inr"] = clean_df["declared_value"].str.replace('₹', '', regex=False).str.replace(',', '', regex=False).astype(int)
  clean_df = clean_df.drop(columns="declared_value")
  clean_df["dispatch_time"] = pd.to_datetime(clean_df["dispatch_time"])
  clean_df.loc[clean_df['parcel_weight_kg'] > 100, 'parcel_weight_kg'] = np.nan
  if medians is None:
    clean_df["distance_km"] = clean_df["distance_km"].fillna(clean_df["distance_km"].median())
    clean_df["parcel_weight_kg"] = clean_df["parcel_weight_kg"].fillna(clean_df["parcel_weight_kg"].median())
    clean_df["declared_value_inr"] = clean_df["declared_value_inr"].fillna(clean_df["declared_value_inr"].median())
  else:
    clean_df["distance_km"] = clean_df["distance_km"].fillna(medians["distance_km"])
    clean_df["parcel_weight_kg"] = clean_df["parcel_weight_kg"].fillna(medians["parcel_weight_kg"])
    clean_df["declared_value_inr"] = clean_df["declared_value_inr"].fillna(medians["declared_value_inr"])
  return (clean_df, medians)
def build_features(clean, hubs):
    df_cleaned = pd.merge(clean, hubs, on="hub_id", how="left")
    df_cleaned["promised_hours"] = df_cleaned["service_type"].map(PROMISED_HOURS)
    df_cleaned["is_cod"] = (df_cleaned["payment_mode"] == "COD").astype(int)
    df_cleaned["is_weekend"] = df_cleaned["dispatch_time"].dt.dayofweek.isin([5, 6]).astype(int)
    df_cleaned["dispatch_hour"] = df_cleaned["dispatch_time"].dt.hour
    df_cleaned["log_declared_value"] = np.log(df_cleaned["declared_value_inr"])
    return df_cleaned
