from src.shipment_prep import clean_shipments, build_features
clean = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")
cleaned_shipments = clean_shipments(clean)
built_features = build_features(cleaned_shipments, hubs)
built_features["is_delayed"] = np.where(df_features["delivery_hours"] > df_features["promised_hours"], 1, 0)
built_features.to_csv("data/modelling_table.csv", index=False)
