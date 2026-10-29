import pandas as pd
from src.shipment_prep import FEATURES
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from slearn.pipeline import Pipeline
modelling_table = pd.read_csv("data/modelling_table.csv")
X = FEATURES
y = modelling_table["delivery_hours"]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_temp = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
checks_df = pd.DataFrame({
  "train_rows": len(X_train_scaled),
  "val_rows": len(X_val),
  "test_rows": len(X_test),
  "train_scaled_max_abs_mean": np.abs(np.mean(X_train_scaled, axis=0)).max(),
  "test_scaled_max_abs_mean": np.abs(np.mean(X_test_scaled, axis=0)).max()
})
checks_df.to_csv("outputs/part_b_checks.csv", index=False)
train_mean = y_train.mean()
y_val_pred_baseline = np.full(shape=len(y_val), fill_value=train_mean)
depths = [1, 2, 3, 4, 6, 8, 12, None]
results = []
best_dtr = None
best_val_rmse = float("inf")
for d in depths:
  dt = DecisionTreeRegressor(max_depth=d, random_state=42)
  dt.fit(X_train, y_train)
  train_rmse = root_mean_squared_error(y_train, dt.predict(X_train))
  val_rmse = root_mean_squared_error(y_val, dt.predict(X_val))
  if val_rmse < best_val_rmse:
    best_val_rmse = val_rmse
    best_dtr = dt
  results.append({
      "max_depth": "unlimited" if d is None else d,
      "train_rmse": round(train_rmse, 4),
      "val_rmse": round(val_rmse, 4),
  })
lowest_val_rmse = df_depth["val_rmse"].min()
df_depth = pd.DataFrame(results)
df_depth.to_csv("outputs/part_b_tree_depth.csv", index=False)
model_metrics = []
model_metrics.append({
    "model": "mean_baseline",
    "split": "val",
    "MAE": round(mean_absolute_error(y_val, y_val_pred_baseline), 4),
    "MSE": round(mean_squared_error(y_val, y_val_pred_baseline), 4),
    "RMSE": round(root_mean_squared_error(y_val, y_val_pred_baseline), 4),
    "R2": round(r2_score(y_val, y_val_pred_baseline, 4),
})
models = {
  "Linear Regression": Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
  ]),
  "Decision Tree Regressor": DecisionTreeRegressor(max_depth=best_dtr.max_depth, random_state=42),
  "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
  "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
}
for name, model in models.items():
  model.fit(X_train, y_train)
  val_preds = model.predict(X_val)
  model_metrics.append({
    "model": name,
    "split": "val",
    "MAE": round(mean_absolute_error(y_val, val_preds), 4)
    "MSE": round(mean_squared_error(y_val, val_preds), 4)
    "RMSE": round(root_mean_squared_error(y_val, val_preds), 4)
    "R2": round(r2_score(y_val, val_preds), 4)
  })
df_metrics = pd.DataFrame(model_metrics)
gb_results = []
lr = [0.01, 0.1, 1.0]
for i in lr:
  GBR = GradientBoostingRegressor(n_estimators=100, learning_rate=i, random_state=42).fit(X_train, y_train)
  val_preds = GBR.predict(X_val)
  gb_results.append({
    "learning_rate": lr,
    "val_rmse": round(root_mean_squared_error(y_val, val_preds), 4)
    "val_r2": round(r2_score(y_val, val_preds), 4)
  })
gb_df = pd.DataFrame(gb_results)
gb_df.to_csv("outputs/part_b_gb_learning_rate.csv", index=False)
non_baseline = df_metrics[df_metrics["model"] != "mean_baseline"]
ranked_candidates = four_candidates.sort_values(by="RMSE")
top_2_names = ranked_candidates["model"].head(2).tolist()
best_candidate_name = top_2_names[0]
X_train_val = pd.concat([X_train, X_val], axis=0)
y_train_val = pd.concat([y_train, y_val], axis=0)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []
for model_name in top_2_names:
  model_obj = models[model_name]
  scores = cross_val_score(
    estimator=model_obj,
    X=X_train_Val,
    y=y_train_val,
    cv=kf
  )
  cv_results.append({
    "model": model_name,
    "mean_r2": round(scores.mean()), 4),
    "std_r2": round(scores.std()), 4),
  })
df_cv = pd.DataFrame(cv_results)
df_cv.to_csv("outputs/part_b_cv.csv", index=False)
final_model = models[best_candidate_name]
test_preds = final_model.predict(X_test)
test_row = {
  "model": best_candidate_name, 
  "split": "test", 
  "MAE": round(mean_absolute_error(y_test, test_preds), 4), 
  "MSE": round(mean_squared_error(y_test, test_preds), 4), 
  "RMSE": round(root_mean_squared_error(y_test, test_preds), 4), 
  "R2": round(r2_score(y_test, test_preds, 4)
  }
df_metrics = pd.concat([df_metrics, pd.DataFrame([test_row])], ignore_index=True)
df_metrics.to_csv("outputs/part_b_metrics.csv", index=False)
