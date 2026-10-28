"""Part C - delay classifier and alert threshold."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from src.shipment_prep import FEATURES, NOISE_COLUMNS

table = pd.read_csv("data/modelling_table.csv")
X, y = table[FEATURES], table["is_delayed"]

# 1. stratified split
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)
print("sizes", len(X_train), len(X_val), len(X_test), "delayed", y_train.sum(), y_val.sum(), y_test.sum())


def cm_row(y_true, pred):
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    n = tn + fp + fn + tp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp), accuracy=(tn + tp) / n,
                precision=prec, recall=rec, f1=f1)


# 2. baseline + three classifiers (threshold 0.5)
rows = [dict(model="blind_all_on_time", **cm_row(y_val, np.zeros(len(y_val), dtype=int)))]
lr = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())]).fit(X_train, y_train)
rows.append(dict(model="logistic_regression", **cm_row(y_val, lr.predict(X_val))))
best_d, best_f1 = None, -1
for d in [2, 3, 4, 5, 6, 8, 10]:
    t = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42).fit(X_train, y_train)
    f = f1_score(y_val, t.predict(X_val))
    if f > best_f1 + 1e-12:   # strict improvement only -> ties keep the smaller depth
        best_d, best_f1 = d, f
tree = DecisionTreeClassifier(criterion="gini", max_depth=best_d, random_state=42).fit(X_train, y_train)
rows.append(dict(model=f"decision_tree_depth_{best_d}", **cm_row(y_val, tree.predict(X_val))))
rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
rows.append(dict(model="random_forest_default", **cm_row(y_val, rf.predict(X_val))))
pd.DataFrame(rows).round(4).to_csv("outputs/part_c_models.csv", index=False)

# 3. which columns carry signal
sel_all = SelectKBest(f_classif, k="all").fit(X_train, y_train)
fs = pd.DataFrame({"feature": FEATURES, "f_score": sel_all.scores_, "rf_importance": rf.feature_importances_})
fs["f_rank"] = fs["f_score"].rank(ascending=False, method="first").astype(int)
fs["is_noise"] = fs["feature"].isin(NOISE_COLUMNS)
fs = fs[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]].sort_values("f_rank")
fs.round(4).to_csv("outputs/part_c_feature_scores.csv", index=False)

k_rows = []
for k in [5, 7, 10, 12, 17]:
    p = Pipeline([("selector", SelectKBest(f_classif, k=k)),
                  ("model", RandomForestClassifier(n_estimators=100, random_state=42))]).fit(X_train, y_train)
    k_rows.append(dict(k=k, val_f1=f1_score(y_val, p.predict(X_val))))
pd.DataFrame(k_rows).round(4).to_csv("outputs/part_c_selectk.csv", index=False)

# 4. tuning with k = 7 (selector fitted on training rows only)
selector = SelectKBest(f_classif, k=7).fit(X_train, y_train)
Xtr7, Xva7 = selector.transform(X_train), selector.transform(X_val)
grid_params = {"n_estimators": [50, 100, 150], "max_depth": [4, 6, 8], "min_samples_leaf": [1, 5, 10]}
gs = GridSearchCV(RandomForestClassifier(random_state=42), grid_params, cv=5).fit(Xtr7, y_train)
rs = RandomizedSearchCV(RandomForestClassifier(random_state=42), grid_params, n_iter=9, cv=5,
                        random_state=42).fit(Xtr7, y_train)
default7 = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr7, y_train)
tuning = [
    dict(search="grid_search", candidates=27, cv_folds=5, total_fits=27 * 5, best_params=str(gs.best_params_),
         best_cv_accuracy=gs.best_score_, val_f1=f1_score(y_val, gs.best_estimator_.predict(Xva7)), _p=gs.best_params_),
    dict(search="randomized_search", candidates=9, cv_folds=5, total_fits=9 * 5, best_params=str(rs.best_params_),
         best_cv_accuracy=rs.best_score_, val_f1=f1_score(y_val, rs.best_estimator_.predict(Xva7)), _p=rs.best_params_),
    dict(search="default_forest_same_7_features", candidates=1, cv_folds=0, total_fits=1,
         best_params=str({"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}),
         best_cv_accuracy=np.nan, val_f1=f1_score(y_val, default7.predict(Xva7)),
         _p={"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}),
]
tun_df = pd.DataFrame(tuning)

# 5. final configuration: highest val_f1, ties -> default, grid, randomized
order = ["default_forest_same_7_features", "grid_search", "randomized_search"]
best_val = tun_df["val_f1"].round(12).max()
winner = next(n for n in order if round(tun_df.set_index("search").loc[n, "val_f1"], 12) == best_val)
params = tun_df.set_index("search").loc[winner, "_p"]
tun_df.drop(columns="_p").round(4).to_csv("outputs/part_c_tuning.csv", index=False)

final = Pipeline([("selector", SelectKBest(f_classif, k=7)),
                  ("model", RandomForestClassifier(random_state=42, **params))]).fit(X_train, y_train)

# 6. alert threshold on validation
proba_val = final.predict_proba(X_val)[:, 1]
th_rows = []
for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    th_rows.append(dict(threshold=t, **cm_row(y_val, (proba_val >= t).astype(int))))
th_df = pd.DataFrame(th_rows)
ok = th_df[th_df["recall"].round(12) >= 0.80]
threshold = float(ok["threshold"].max()) if len(ok) else 0.1
th_df.round(4).to_csv("outputs/part_c_thresholds.csv", index=False)

# 7. one test evaluation
proba_test = final.predict_proba(X_test)[:, 1]
pd.DataFrame([dict(threshold=threshold, **cm_row(y_test, (proba_test >= threshold).astype(int)))]) \
    .round(4).to_csv("outputs/part_c_final_test.csv", index=False)
md = params["max_depth"]
pd.DataFrame([dict(selector_k=7, n_estimators=params["n_estimators"],
                   max_depth="unlimited" if md is None else int(md),
                   min_samples_leaf=params["min_samples_leaf"], threshold=threshold, chosen_from=winner)]) \
    .to_csv("outputs/part_c_final_config.csv", index=False)

for f in ["part_c_models", "part_c_feature_scores", "part_c_selectk", "part_c_tuning", "part_c_thresholds",
          "part_c_final_test", "part_c_final_config"]:
    print(f); print(pd.read_csv(f"outputs/{f}.csv").to_string(index=False))
print("noise importance share:", fs.loc[fs["is_noise"], "rf_importance"].sum())
