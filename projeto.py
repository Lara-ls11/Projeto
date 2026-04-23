import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


RSS_NO_SIGNAL = 100
RSS_FILL_VALUE = -110.0
RANDOM_STATE = 42


def load_datasets():
    print("A carregar dados...")
    train_x = pd.read_csv("ipin2022_trainrss.csv")
    train_y = pd.read_csv("ipin2022_trainflr.csv").iloc[:, 0]

    val_x = pd.read_csv("ipin2022_validrss.csv")
    val_y = pd.read_csv("ipin2022_validflr.csv").iloc[:, 0]

    test_x = pd.read_csv("ipin2022_testrss.csv")
    test_y = pd.read_csv("ipin2022_testsflr.csv").iloc[:, 0]
    return train_x, train_y, val_x, val_y, test_x, test_y


def align_columns(train_x, val_x, test_x):
    all_columns = sorted(set(train_x.columns) | set(val_x.columns) | set(test_x.columns))
    train_x = train_x.reindex(columns=all_columns, fill_value=np.nan)
    val_x = val_x.reindex(columns=all_columns, fill_value=np.nan)
    test_x = test_x.reindex(columns=all_columns, fill_value=np.nan)
    return train_x, val_x, test_x


def replace_no_signal(df):
    df = df.copy()
    df.replace(RSS_NO_SIGNAL, np.nan, inplace=True)
    return df


def add_rss_features(df):
    df = df.copy()
    values = df.to_numpy(dtype=float)
    valid_mask = ~np.isnan(values)
    strong_thresholds = (-85, -75, -65)

    safe_values = np.where(valid_mask, values, np.nan)
    observed_count = valid_mask.sum(axis=1)
    df["ap_count"] = observed_count
    df["missing_ratio"] = 1.0 - (observed_count / max(df.shape[1], 1))

    df["rss_mean"] = np.nanmean(safe_values, axis=1)
    df["rss_std"] = np.nanstd(safe_values, axis=1)
    df["rss_max"] = np.nanmax(safe_values, axis=1)
    df["rss_min"] = np.nanmin(safe_values, axis=1)
    df["rss_median"] = np.nanmedian(safe_values, axis=1)
    df["rss_range"] = df["rss_max"] - df["rss_min"]

    for q in (10, 25, 75, 90):
        df[f"rss_q{q}"] = np.nanpercentile(safe_values, q, axis=1)

    sorted_desc = np.sort(np.where(valid_mask, values, -999.0), axis=1)[:, ::-1]
    for top_k in (3, 5, 10):
        clipped = sorted_desc[:, :top_k]
        valid_top = clipped > -999.0
        df[f"top{top_k}_mean"] = np.where(
            valid_top.any(axis=1),
            np.nanmean(np.where(valid_top, clipped, np.nan), axis=1),
            RSS_FILL_VALUE,
        )

    for threshold in strong_thresholds:
        df[f"count_gt_{abs(threshold)}"] = np.nansum(safe_values > threshold, axis=1)

    return df.fillna(RSS_FILL_VALUE)


def build_search_spaces():
    models = {
        "KNN": (
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="constant", fill_value=RSS_FILL_VALUE)),
                    ("scaler", StandardScaler()),
                    ("selector", SelectKBest(score_func=mutual_info_classif)),
                    ("clf", KNeighborsClassifier()),
                ]
            ),
            {
                "selector__k": [60, 90, 120, 180],
                "clf__n_neighbors": [3, 5, 7, 9],
                "clf__weights": ["distance"],
                "clf__metric": ["manhattan", "euclidean"],
            },
        ),
        "SVM": (
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="constant", fill_value=RSS_FILL_VALUE)),
                    ("scaler", StandardScaler()),
                    ("selector", SelectKBest(score_func=mutual_info_classif)),
                    ("clf", SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE)),
                ]
            ),
            {
                "selector__k": [90, 120, 180, "all"],
                "clf__C": [3, 10, 30],
                "clf__gamma": ["scale", 0.01, 0.03],
                "clf__kernel": ["rbf"],
            },
        ),
        "Random Forest": (
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="constant", fill_value=RSS_FILL_VALUE)),
                    ("selector", SelectKBest(score_func=mutual_info_classif)),
                    (
                        "clf",
                        RandomForestClassifier(
                            random_state=RANDOM_STATE,
                            class_weight="balanced_subsample",
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            {
                "selector__k": [90, 120, 180, "all"],
                "clf__n_estimators": [400, 700],
                "clf__max_depth": [None, 20, 35],
                "clf__min_samples_leaf": [1, 2, 4],
                "clf__max_features": ["sqrt", 0.35, 0.5],
            },
        ),
        "Extra Trees": (
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="constant", fill_value=RSS_FILL_VALUE)),
                    ("selector", SelectKBest(score_func=mutual_info_classif)),
                    (
                        "clf",
                        ExtraTreesClassifier(
                            random_state=RANDOM_STATE,
                            class_weight="balanced",
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            {
                "selector__k": [90, 120, 180, "all"],
                "clf__n_estimators": [500, 800],
                "clf__max_depth": [None, 25, 40],
                "clf__min_samples_leaf": [1, 2],
                "clf__max_features": ["sqrt", 0.35, 0.5],
            },
        ),
    }
    return models


def run_model_selection(train_x, train_y, val_x, val_y):
    x_search = pd.concat([train_x, val_x], ignore_index=True)
    y_search = pd.concat([train_y, val_y], ignore_index=True)
    split_index = np.concatenate(
        [
            np.full(len(train_x), -1, dtype=int),
            np.zeros(len(val_x), dtype=int),
        ]
    )
    predefined_split = PredefinedSplit(test_fold=split_index)

    best_models = {}
    validation_scores = {}

    for name, (pipeline, params) in build_search_spaces().items():
        print(f"\nA otimizar {name}...")
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=params,
            scoring="accuracy",
            cv=predefined_split,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(x_search, y_search)
        best_models[name] = search.best_estimator_
        validation_scores[name] = search.best_score_
        print(f"  Melhor accuracy de validação: {search.best_score_:.4f}")
        print(f"  Melhores parâmetros: {search.best_params_}")

    return best_models, validation_scores


def build_weighted_ensemble(best_models, validation_scores):
    estimators = [(name, model) for name, model in best_models.items()]
    weights = [max(1, int(round(validation_scores[name] * 100))) for name in best_models]
    ensemble = VotingClassifier(
        estimators=estimators,
        voting="soft",
        weights=weights,
        n_jobs=-1,
    )
    return ensemble, weights


def evaluate_model(name, model, x, y, encoder, dataset_name):
    pred_encoded = model.predict(x)
    pred = encoder.inverse_transform(pred_encoded)
    acc = accuracy_score(y, pred)
    print(f"\n{'-' * 60}")
    print(f"Modelo   : {name}")
    print(f"Conjunto : {dataset_name}")
    print(f"Accuracy : {acc:.4f} ({acc * 100:.2f}%)")
    print(confusion_matrix(y, pred))
    print(classification_report(y, pred))
    return acc


def main():
    train_x, train_y_raw, val_x, val_y_raw, test_x, test_y_raw = load_datasets()
    train_x, val_x, test_x = align_columns(train_x, val_x, test_x)

    train_x = replace_no_signal(train_x)
    val_x = replace_no_signal(val_x)
    test_x = replace_no_signal(test_x)

    train_x = add_rss_features(train_x)
    val_x = add_rss_features(val_x)
    test_x = add_rss_features(test_x)

    encoder = LabelEncoder()
    train_y = pd.Series(encoder.fit_transform(train_y_raw), index=train_y_raw.index)
    val_y = pd.Series(encoder.transform(val_y_raw), index=val_y_raw.index)
    best_models, validation_scores = run_model_selection(train_x, train_y, val_x, val_y)

    print("\nResumo da validação:")
    for name, score in sorted(validation_scores.items(), key=lambda item: item[1], reverse=True):
        print(f"  {name:<14} -> {score:.4f}")

    ensemble, weights = build_weighted_ensemble(best_models, validation_scores)
    print(f"\nA construir ensemble probabilístico com pesos {weights}...")

    full_train_x = pd.concat([train_x, val_x], ignore_index=True)
    full_train_y = pd.concat([train_y, val_y], ignore_index=True)

    final_models = {}
    for name, model in best_models.items():
        model.fit(full_train_x, full_train_y)
        final_models[name] = model

    ensemble.fit(full_train_x, full_train_y)
    final_models["Ensemble"] = ensemble

    print("\n" + "=" * 60)
    print("RESULTADOS NO TESTE FINAL")
    print("=" * 60)

    best_name = None
    best_model = None
    best_acc = -1.0

    for name, model in final_models.items():
        acc = evaluate_model(name, model, test_x, test_y_raw, encoder, "Teste")
        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_model = model

    print("\n" + "=" * 60)
    print(f"MELHOR MODELO: {best_name}")
    print(f"Accuracy no teste: {best_acc * 100:.2f}%")
    print("=" * 60)

    return best_model


if __name__ == "__main__":
    main()
