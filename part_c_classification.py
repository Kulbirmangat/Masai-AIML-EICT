import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from src.shipment_prep import FEATURES, NOISE_COLUMNS
def compute_metrics_from_cm(tn, fp, fn, tp, force_precision_zero=False):
  total = tn + fp + fn + tp
  accuracy = (tp + tn) / total if total > 0 else 0.0
  if force_precision_zero:
    precision = 0.0
  else:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
  recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
  f1 = ((2 * precision * recall) / (precision + recall)
      if (precision + recall) > 0 
      else 0.0)
  return (
      round(accuracy, 4),
      round(precision, 4),
      round(recall, 4),
      round(f1, 4),
  )
modelling_table = pd.read_csv("data/modelling_table.csv")
X = modelling_table[FEATURES]
y = modelling_table["is_delayed"]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
models_list = []
y_val_pred_baseline = np.zeros(len(y_val), dtype=int)
TN, FP, FN, TP = confusion_matrix(y_val, y_val_pred_baseline, labels=[0, 1]).ravel()
acc, prec, rec, f1 = compute_metrics_from_cm(TN, FP, FN, TP, force_precision_zero=True)
models_list.append({
    "model": "blind_all_on_time",
    "TN": int(TN),
    "FP": int(FP),
    "FN": int(FN),
    "TP": int(TP),
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1": f1,
})
depths = [2, 3, 4, 5, 6, 8, 10]
d, d_f1 = 0, float("-inf")
for i in depths:
  DTC = DecisionTreeClassifier(
      criterion="gini", max_depth=i, random_state=42
  ).fit(X_train, y_train)
  DTC_preds = DTC.predict(X_val)
  tn, fp, fn, tp = confusion_matrix(y_val, DTC_preds, labels=[0, 1]).ravel()
  _, _, _, DTC_f1 = compute_metrics_from_cm(tn, fp, fn, tp)
  if DTC_f1 > d_f1:
    d = i
    d_f1 = DTC_f1
models = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()), 
        ("model", LogisticRegression())
    )],
    f"decision_tree_depth_{d}": DecisionTreeClassifier(
        criterion="gini", max_depth=d, random_state=42
    ),
    "random_forest_default": RandomForestClassifier(
        n_estimators=100, random_state=42
    )}
for name, model in models.items():
  model.fit(X_train, y_train)
  model_preds = model.predict(X_val)
  TN, FP, FN, TP = confusion_matrix(y_val, model_preds, labels=[0, 1]).ravel()
  acc, prec, rec, f1 = compute_metrics_from_cm(TN, FP, FN, TP)
  models_list.append({
      "model": name,
      "TN": int(TN),
      "FP": int(FP),
      "FN": int(FN),
      "TP": int(TP),
      "accuracy": acc,
      "precision": prec,
      "recall": rec,
      "f1": f1,
  })
df_models = pd.DataFrame(models_list)
df_models.to_csv("outputs/part_c_models.csv", index=False)
default_forest = RandomForestClasssifier(n_estimators=100, random_state=42).fit(X_train, y_train)
default_forest_importances = default_forest.feature_importances_
select_training = SelectKBest(f_classif,k="all").fit(X_train, y_train)
f_scores = select_training.scores_
feature_names = list(X_train.columns)
df_features = pd.DataFrame({
        "feature": feature_names,
        "f_score": f_scores,
        "rf_importance": rf_importances,
    })
df_features["f_rank"] = (df_features["f_score"].rank(ascending=False, method="min").astype(int))
df_features["is_noise"] = df_features["feature"].isin(NOISE_COLUMNS)
df_features = df_features[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]]
df_features.to_csv("outputs/part_c_feature_scores.csv", index=False)
k_candidates = [5, 7, 10, 12, 17]
selectk_results = []
for k in k_candidates:
  pipe = Pipeline([
          ("selectk", SelectKBest(score_func=f_classif, k=k)),
          ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
      ])
  pipe.fit(X_train, y_train)
  y_pred_val = pipe.predict(X_val)
  val_f1 = f1_score(y_val, y_pred_val, zero_division=0)
  selectk_results.append({"k": k, "val_f1": round(val_f1, 4)})
df_selectk = pd.DataFrame(selectk_results)
df_selectk.to_csv("outputs/part_c_selectk.csv", index=False)
select_k_7 = SelectKBest(f_classif, k=7).fit(X_train, y_train)
X_train_select = select_k_7.fit_transform(X_train)
X_val_select = select_k_7.fit(X_val)
tuning_rows = []
param_grid = {
  "n_estimators": [50, 100, 150],
  "max_depth": [4, 6, 8],
  "min_samples_leaf": [1, 5, 10],
}
rf_base = RandomForestClassifier(random_state=42).fit(X_train_select, y_train)
grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, cv=5).fit(X_train, y_train)
y_pred_grid = grid_search.best_estimator_.predict(X_val_select)
tuning_rows.append({
  "search": "grid_search",
  "candidates": len(grid_search.cv_results_["params"]),
  "cv_folds": 5,
  "total_fits": len(grid_search.cv_results_["params"]) * 5,
  "best_params": str(grid_search.best_params_),
  "best_cv_accuracy": round(grid_search.best_score_, 4)
  "val_f1": round(f1_score(y_val, y_pred_grid, zero_division=0), 4)
})
random_search = RandomizedSearchCV(estimator=rf_base, param_distributions=param_grid, n_iter=9, cv=5, random_state=42).fit(X_train, y_train)
y_pred_random = random_search.best_estimator_.predict(X_val_select)
tuning_rows.append({
  "search": "randomized_search",
  "candidates": len(random_search.cv_results_["params"]),
  "cv_folds": 5,
  "total_fits": len(random_search.cv_results_["params"]) * 5,
  "best_params": str(random_search.best_params_),
  "best_cv_accuracy": round(random_search.best_score_, 4)
  "val_f1": round(f1_score(y_val, y_pred_random, zero_division=0), 4)
})
rfc = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_leaf=1, random_state=42).fit(X_train, y_train)
y_pred_rfc = rfc.predict(X_val_select)
tuning_rows.append({
  "search": "default_forest_same_7_features",
  "candidates": 1,
  "cv_folds": 0,
  "total_fits": 1,
  "best_params": "defaults",
  "best_cv_accuracy": "",
  "val_f1": round(f1_score(y_val, y_pred_rfc), 4)
})
tuning_df = pd.DataFrame(tuning_rows)
tuning_df.to_csv("outputs/part_c_tuning.csv", index=False)
priority = {"default_forest_same_7_features": 0, "grid_search": 1, "randomized_search": 2}
best_row = max(tuning_rows, key=lambda row: (row["val_f1"], -priority[row["search"]]))
chosen_params = best_row["best_params"] if isinstance(best_row["best_params"], dict) else {}
final_model = Pipeline([
    ("selector", SelectKBest(score_func=f_classif, k=7)),
    ("model", RandomForestClassifier(random_state=42, **chosen_params)),
]).fit(X_train, y_train)
probability = final_model.predict_proba(X_val)[:, 1]
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
threshold_list = []
for t in thresholds:
  predictions_array = np.where(probability >= t, "delayed", "not delayed")
  TN, FP, FN, TP = confusion_matrix(y_val, predictions_array, labels=["not delayed", "delayed"]).ravel()
  accuracy, precision, recall, f1 = compute_metrics_from_cm(TN, FP, FN, TP)
  threshold_list.append({
    "threshold": t,
    "TN": int(TN),
    "FP": int(FP),
    "FN": int(FN),
    "TP": int(TP),
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1
  })
df_threshold = pd.DataFrame(threshold_list)
df_threshold.to_csv("outputs/part_c_thresholds.csv", index=False)
valid_thresholds = df_thresh[df_thresh["recall"] >= 0.80]["threshold"]
alert_threshold = valid_thresholds.max() if not valid_thresholds.empty else 0.1
test_probability = final_model.predict_proba(X_test)[:, 1]
test_predictions_array = np.where(test_probability >= alert_threshold, "delayed", "not delayed")
TN, FP, FN, TP = confusion_matrix(y_test, test_predictions_array, labels=["not delayed", "delayed"]).ravel()
accuracy, precision, recall, f1 = compute_metrics_from_cm(TN, FP, FN, TP)
df_final_test = pd.DataFrame([{
    "threshold": alert_threshold,
    "TN": int(TN),
    "FP": int(FP),
    "FN": int(FN),
    "TP": int(TP),
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1
}])
df_final_test.to_csv("outputs/part_c_final_test.csv", index=False)
rf_model = final_model.named_steps["model"]
raw_depth = rf_model.max_depth
config_row = {
    "selector_k": final_model.named_steps["selector"].k,
    "n_estimators": rf_model.n_estimators,
    "max_depth": "unlimited" if raw_depth is None else raw_depth,
    "min_samples_leaf": rf_model.min_samples_leaf,
    "threshold": alert_threshold,
    "chosen_from": best_row["search"] if "best_row" in locals() and isinstance(best_row, dict) and "search" in best_row else pd.read_csv("outputs/part_c_tuning.csv").sort_values(by=["val_f1", "search"], ascending=[False, True]).iloc[0]["search"]
}
config_df = pd.DataFrame([config_row])
config_df.to_csv("outputs/part_c_final_config.csv", index=False)
