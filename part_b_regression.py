"""Part B - delivery-hours regression."""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from src.shipment_prep import FEATURES

table = pd.read_csv("data/modelling_table.csv")
X, y = table[FEATURES], table["delivery_hours"]

# 1. split
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

# 2. scaling without leakage
scaler = StandardScaler().fit(X_train)
tr_scaled, te_scaled = scaler.transform(X_train), scaler.transform(X_test)
pd.DataFrame({"metric": ["train_rows", "val_rows", "test_rows", "train_scaled_max_abs_mean", "test_scaled_max_abs_mean"],
              "value": [len(X_train), len(X_val), len(X_test),
                        np.abs(tr_scaled.mean(axis=0)).max(), np.abs(te_scaled.mean(axis=0)).max()]}
             ).round(4).to_csv("outputs/part_b_checks.csv", index=False)
# the 4-dp rounding would show 0.0 for the train mean, which is what we want to prove (< 1e-9)
print("train max abs mean (unrounded):", np.abs(tr_scaled.mean(axis=0)).max())


def metrics(y_true, pred):
    mse = mean_squared_error(y_true, pred)
    return dict(MAE=mean_absolute_error(y_true, pred), MSE=mse, RMSE=float(np.sqrt(mse)), R2=r2_score(y_true, pred))


rows = []
# 3. baseline
rows.append(dict(model="mean_baseline", split="val",
                 **metrics(y_val, np.full(len(y_val), y_train.mean()))))

# 4. tree-depth diagnostic
depth_rows = []
for d in [1, 2, 3, 4, 6, 8, 12, None]:
    t = DecisionTreeRegressor(max_depth=d, random_state=42).fit(X_train, y_train)
    depth_rows.append(dict(max_depth="unlimited" if d is None else d,
                           train_rmse=np.sqrt(mean_squared_error(y_train, t.predict(X_train))),
                           val_rmse=np.sqrt(mean_squared_error(y_val, t.predict(X_val))), _d=d))
depth_df = pd.DataFrame(depth_rows)
best_i = int(depth_df["val_rmse"].round(12).idxmin())  # first (= smaller depth) among ties
best_depth = depth_df.loc[best_i, "_d"]
best_depth = None if pd.isna(best_depth) else int(best_depth)
depth_df.drop(columns="_d").round(4).to_csv("outputs/part_b_tree_depth.csv", index=False)

# 5. candidate models
candidates = {
    "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
}
for name, m in candidates.items():
    m.fit(X_train, y_train)
    rows.append(dict(model=name, split="val", **metrics(y_val, m.predict(X_val))))

# 6. learning-rate experiment
lr_rows = []
for lr in [0.01, 0.1, 1.0]:
    g = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42).fit(X_train, y_train)
    p = g.predict(X_val)
    lr_rows.append(dict(learning_rate=lr, val_rmse=np.sqrt(mean_squared_error(y_val, p)), val_r2=r2_score(y_val, p)))
pd.DataFrame(lr_rows).round(4).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)

# 7. cross-validation of the two best, final model
val_rows = pd.DataFrame(rows)
ranked = val_rows[val_rows["model"] != "mean_baseline"].sort_values("RMSE")
top2 = list(ranked["model"].head(2))
kf = KFold(n_splits=5, shuffle=True, random_state=42)
X_trval, y_trval = pd.concat([X_train, X_val]), pd.concat([y_train, y_val])
cv_rows = []
for name in top2:
    s = cross_val_score(candidates[name], X_trval, y_trval, cv=kf)
    cv_rows.append(dict(model=name, mean_r2=s.mean(), std_r2=s.std()))
pd.DataFrame(cv_rows).round(4).to_csv("outputs/part_b_cv.csv", index=False)

final_name = top2[0]
rows.append(dict(model=final_name, split="test", **metrics(y_test, candidates[final_name].predict(X_test))))
pd.DataFrame(rows).round(4).to_csv("outputs/part_b_metrics.csv", index=False)

for f in ["part_b_checks", "part_b_metrics", "part_b_tree_depth", "part_b_gb_learning_rate", "part_b_cv"]:
    print(f); print(pd.read_csv(f"outputs/{f}.csv").to_string(index=False))
print("best depth:", best_depth, "final:", final_name)
