# IMPORTAR BIBLIOTECAS


import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import VarianceThreshold
import warnings
warnings.filterwarnings("ignore")

# CARREGAR DATASETS


train_rss = pd.read_csv("ipin2022_trainrss.csv")
train_flr = pd.read_csv("ipin2022_trainflr.csv")
valid_rss = pd.read_csv("ipin2022_validrss.csv")
valid_flr = pd.read_csv("ipin2022_validflr.csv")
test_rss  = pd.read_csv("ipin2022_testrss.csv")
test_flr  = pd.read_csv("ipin2022_testflr.csv")


# DIAGNÓSTICO INICIAL


label_column = "0"

print("===== DIAGNÓSTICO =====")
print(f"Train samples: {len(train_rss)}")
print(f"Valid samples: {len(valid_rss)}")
print(f"Test  samples: {len(test_rss)}")
print(f"\nClasses (train): {sorted(train_flr[label_column].unique())}")
print(f"Distribuição train:\n{train_flr[label_column].value_counts().sort_index()}")
print(f"\nDistribuição valid:\n{valid_flr[label_column].value_counts().sort_index()}")


# ALINHAR COLUNAS


all_columns = sorted(set(train_rss.columns) | set(valid_rss.columns) | set(test_rss.columns))
NO_SIGNAL   = -110

train_rss = train_rss.reindex(columns=all_columns, fill_value=NO_SIGNAL)
valid_rss  = valid_rss.reindex(columns=all_columns,  fill_value=NO_SIGNAL)
test_rss   = test_rss.reindex(columns=all_columns,   fill_value=NO_SIGNAL)

print(f"\nTotal de APs (features brutas): {len(all_columns)}")


# PRÉ-PROCESSAMENTO 


def preprocess_rss(df, no_signal=NO_SIGNAL, threshold=-95):
    df = df.copy()
    df[df < threshold] = no_signal
    return df

train_rss = preprocess_rss(train_rss)
valid_rss  = preprocess_rss(valid_rss)
test_rss   = preprocess_rss(test_rss)


# FEATURE ENGINEERING


def add_rss_features(rss_df, no_signal=NO_SIGNAL):
    df = rss_df.copy()
    detected = rss_df > no_signal

    rss_det = rss_df.where(detected, other=np.nan)
    df["f_n_detected"]  = detected.sum(axis=1)
    df["f_max_rss"]     = rss_det.max(axis=1).fillna(no_signal)
    df["f_min_rss"]     = rss_det.min(axis=1).fillna(no_signal)
    df["f_mean_rss"]    = rss_det.mean(axis=1).fillna(no_signal)
    df["f_std_rss"]     = rss_det.std(axis=1).fillna(0)
    df["f_range_rss"]   = df["f_max_rss"] - df["f_min_rss"]

    for k in [3, 5, 10]:
        df[f"f_top{k}_mean"] = rss_det.apply(
            lambda row: row.nlargest(k).mean() if row.notna().sum() >= k else row.mean(),
            axis=1
        ).fillna(no_signal)

    df["f_pct_detected"] = df["f_n_detected"] / len(rss_df.columns)

    linear = rss_df.where(detected, other=0).apply(lambda x: 10 ** (x / 10))
    df["f_energy"] = np.log1p(linear.sum(axis=1))

    strong = (rss_df > -70) & detected
    df["f_n_strong"] = strong.sum(axis=1)

    medium = (rss_df > -85) & (rss_df <= -70) & detected
    df["f_n_medium"] = medium.sum(axis=1)

    return df

print("\nA gerar features...")
train_feat = add_rss_features(train_rss)
valid_feat  = add_rss_features(valid_rss)
test_feat   = add_rss_features(test_rss)

# Labels
y_train = train_flr[label_column].values
y_valid  = valid_flr[label_column].values
y_test   = test_flr[label_column].values

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_valid_enc  = le.transform(y_valid)
y_test_enc   = le.transform(y_test)


# REMOVER FEATURES SEM VARIÂNCIA


print("A remover features sem variância...")
selector = VarianceThreshold(threshold=0.01)
X_train_sel = selector.fit_transform(train_feat.fillna(0))
X_valid_sel  = selector.transform(valid_feat.fillna(0))
X_test_sel   = selector.transform(test_feat.fillna(0))
print(f"Features após remoção: {X_train_sel.shape[1]} (eram {train_feat.shape[1]})")


# NORMALIZAÇÃO

scaler = RobustScaler()
X_train_sc = scaler.fit_transform(X_train_sel)
X_valid_sc  = scaler.transform(X_valid_sel)
X_test_sc   = scaler.transform(X_test_sel)


# TREINO + VALIDAÇÃO


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- KNN ---
print("\nA otimizar KNN...")
best_k, best_metric, best_acc_knn = 1, "manhattan", 0
for metric in ["minkowski", "manhattan", "chebyshev"]:
    for k in [1, 3, 5, 7, 9, 11, 15, 20]:
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric, n_jobs=-1)
        knn.fit(X_train_sc, y_train_enc)
        acc = accuracy_score(y_valid_enc, knn.predict(X_valid_sc))
        if acc > best_acc_knn:
            best_acc_knn, best_k, best_metric = acc, k, metric

knn_best = KNeighborsClassifier(n_neighbors=best_k, metric=best_metric, n_jobs=-1)
knn_best.fit(X_train_sc, y_train_enc)
print(f"  k={best_k}, metric={best_metric}, acc_valid={best_acc_knn:.4f}")

# --- SVM ---
print("A otimizar SVM...")
params_svm = {
    "C":     [1, 10, 50, 100, 200],
    "gamma": ["scale", 0.01, 0.001],
    "kernel":["rbf"]
}
grid_svm = GridSearchCV(
    SVC(probability=True, cache_size=500),
    params_svm, cv=3, n_jobs=-1, verbose=0
)
grid_svm.fit(X_train_sc, y_train_enc)
svm_best = grid_svm.best_estimator_
print(f"  {grid_svm.best_params_}, acc_valid={accuracy_score(y_valid_enc, svm_best.predict(X_valid_sc)):.4f}")

# --- Random Forest ---
print("A treinar Random Forest...")
rf_best = RandomForestClassifier(
    n_estimators=100, max_depth=None,
    min_samples_split=2, min_samples_leaf=1,
    max_features="sqrt", n_jobs=-1, random_state=42
)
rf_best.fit(X_train_sc, y_train_enc)
print(f"  acc_valid={accuracy_score(y_valid_enc, rf_best.predict(X_valid_sc)):.4f}")


# AVALIAR MODELOS


models = [
    ("KNN",           knn_best),
    ("SVM",           svm_best),
    ("Random Forest", rf_best),
]

results = {}
for name, model in models:
    preds = le.inverse_transform(model.predict(X_valid_sc))
    acc   = accuracy_score(y_valid, preds)
    results[name] = acc
    print(f"\n===== {name} — Accuracy: {acc:.4f} =====")
    print(classification_report(y_valid, preds))


# RESULTADO FINAL + TEST SET

results_df = pd.DataFrame.from_dict(results, orient="index", columns=["Accuracy (valid)"])
results_df = results_df.sort_values("Accuracy (valid)", ascending=False)

print("\n\n===== RESULTADOS FINAIS =====")
print(results_df.to_string())

best_name  = results_df.index[0]
best_model = dict(models)[best_name]

test_preds = le.inverse_transform(best_model.predict(X_test_sc))
test_acc   = accuracy_score(y_test, test_preds)

print(f"\n→ Melhor modelo: {best_name}")
print(f"→ Accuracy no TEST SET: {test_acc:.4f}")
print(f"\n{classification_report(y_test, test_preds)}")
