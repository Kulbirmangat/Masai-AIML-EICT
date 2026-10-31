import pandas as pd
import numpy as np
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
