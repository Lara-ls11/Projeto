import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, PredefinedSplit


def run_model_selection(train_x, train_y, val_x, val_y, model_configs):
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

    for name, (pipeline, params) in model_configs.items():
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


def run_single_model_selection(train_x, train_y, val_x, val_y, name, pipeline, params):
    return run_model_selection(
        train_x,
        train_y,
        val_x,
        val_y,
        {name: (pipeline, params)},
    )
