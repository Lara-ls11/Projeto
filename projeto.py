import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. CARREGAR FICHEIROS CSV
# ============================================================
print("A carregar dados...")
train_X = pd.read_csv("ipin2022_trainrss.csv")
train_y = pd.read_csv("ipin2022_trainflr.csv").iloc[:, 0]

val_X   = pd.read_csv("ipin2022_validrss.csv")
val_y   = pd.read_csv("ipin2022_validflr.csv").iloc[:, 0]

test_X  = pd.read_csv("ipin2022_testrss.csv")
test_y  = pd.read_csv("ipin2022_testsflr.csv").iloc[:, 0]

# ============================================================
# 2. ALINHAR COLUNAS (igual para todos os conjuntos)
# ============================================================
all_columns = sorted(set(train_X.columns) | set(val_X.columns) | set(test_X.columns))
train_X = train_X.reindex(columns=all_columns, fill_value=np.nan)
val_X   = val_X.reindex(columns=all_columns,   fill_value=np.nan)
test_X  = test_X.reindex(columns=all_columns,  fill_value=np.nan)

# ============================================================
# 3. PRÉ-PROCESSAMENTO DOS SINAIS RSS
# ============================================================
# Valor típico de "sem sinal" em datasets RSS é 100 ou +100 → substituir por NaN
RSS_NO_SIGNAL = 100  # ajusta se necessário (ex: 0, -999, 100)
for df in [train_X, val_X, test_X]:
    df.replace(RSS_NO_SIGNAL, np.nan, inplace=True)

# Substituir NaN pelo valor mínimo de cada coluna (sinal mais fraco observado)
col_min = train_X.min()                         # aprende apenas no treino
train_X = train_X.fillna(col_min)
val_X   = val_X.fillna(col_min)
test_X  = test_X.fillna(col_min)

# ============================================================
# 4. FEATURE ENGINEERING — estatísticas dos sinais RSS
# ============================================================
def add_rss_features(df):
    df = df.copy()
    values = df.values
    df["rss_mean"]    = np.nanmean(values, axis=1)
    df["rss_std"]     = np.nanstd(values,  axis=1)
    df["rss_max"]     = np.nanmax(values,  axis=1)
    df["rss_min"]     = np.nanmin(values,  axis=1)
    df["rss_range"]   = df["rss_max"] - df["rss_min"]
    # Número de APs com sinal "bom" (ex: RSS > -80 dBm; ajusta ao teu dataset)
    threshold = col_min.max() * 0.7          # heurística relativa
    df["rss_n_strong"] = (df.iloc[:, :-5] > threshold).sum(axis=1)
    return df

train_X = add_rss_features(train_X)
val_X   = add_rss_features(val_X)
test_X  = add_rss_features(test_X)

# ============================================================
# 5. NORMALIZAÇÃO (StandardScaler ajustado só no treino)
# ============================================================
scaler = StandardScaler()
train_X_sc = scaler.fit_transform(train_X)
val_X_sc   = scaler.transform(val_X)
test_X_sc  = scaler.transform(test_X)

# ============================================================
# 6. DEFINIR MODELOS COM HIPERPARÂMETROS MELHORADOS
# ============================================================

# -- KNN optimizado
print("\nA otimizar KNN...")
knn_params = {"n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"], "metric": ["euclidean", "manhattan"]}
knn = GridSearchCV(KNeighborsClassifier(), knn_params, cv=3, scoring="accuracy", n_jobs=-1)
knn.fit(train_X_sc, train_y)
print(f"  Melhores parâmetros KNN: {knn.best_params_}")

# -- SVM optimizado
print("A otimizar SVM...")
svm_params = {"C": [1, 10, 100], "gamma": ["scale", "auto"], "kernel": ["rbf", "poly"]}
svm = GridSearchCV(SVC(probability=True), svm_params, cv=3, scoring="accuracy", n_jobs=-1)
svm.fit(train_X_sc, train_y)
print(f"  Melhores parâmetros SVM: {svm.best_params_}")

# -- Random Forest optimizado
print("A otimizar Random Forest...")
rf_params = {"n_estimators": [200, 400], "max_depth": [None, 20, 40], "min_samples_split": [2, 5], "max_features": ["sqrt", "log2"]}
rf = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3, scoring="accuracy", n_jobs=-1)
rf.fit(train_X_sc, train_y)
print(f"  Melhores parâmetros RF: {rf.best_params_}")

# -- Gradient Boosting
print("A treinar Gradient Boosting...")
gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8, random_state=42)
gb.fit(train_X_sc, train_y)

# ============================================================
# 7. ENSEMBLE — votação entre os melhores modelos
# ============================================================
print("A construir Ensemble (Voting)...")
ensemble = VotingClassifier(
    estimators=[("knn", knn.best_estimator_), ("svm", svm.best_estimator_),
                ("rf",  rf.best_estimator_),  ("gb",  gb)],
    voting="hard"
)
ensemble.fit(train_X_sc, train_y)

# ============================================================
# 8. AVALIAÇÃO
# ============================================================
modelos = [
    ("KNN (otimizado)",      knn.best_estimator_),
    ("SVM (otimizado)",      svm.best_estimator_),
    ("Random Forest (otim)", rf.best_estimator_),
    ("Gradient Boosting",    gb),
    ("Ensemble (Voting)",    ensemble),
]

def avaliar(nome, modelo, X, y, conjunto="Validação"):
    pred = modelo.predict(X)
    acc  = accuracy_score(y, pred)
    print(f"\n{'─'*55}")
    print(f"  Modelo : {nome}  |  Conjunto: {conjunto}")
    print(f"  Acurácia: {acc:.4f} ({acc*100:.2f}%)")
    print(confusion_matrix(y, pred))
    print(classification_report(y, pred))
    return acc

print("\n" + "="*55)
print("              RESULTADOS — VALIDAÇÃO")
print("="*55)
for nome, modelo in modelos:
    avaliar(nome, modelo, val_X_sc, val_y, "Validação")

print("\n" + "="*55)
print("              RESULTADOS — TESTE FINAL")
print("="*55)
best_acc, best_name, best_model = 0, "", None
for nome, modelo in modelos:
    acc = avaliar(nome, modelo, test_X_sc, test_y, "Teste")
    if acc > best_acc:
        best_acc, best_name, best_model = acc, nome, modelo

print("\n" + "="*55)
print(f"  🏆 MELHOR MODELO: {best_name}")
print(f"     Acurácia no teste: {best_acc*100:.2f}%")
print("="*55)