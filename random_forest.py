import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from common import RANDOM_STATE, RSS_FILL_VALUE
from train_model import train_and_evaluate

warnings.filterwarnings("ignore")

NAME = "Random Forest"


def get_search_config():
    pipeline = Pipeline(
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
    )
    params = {
        "selector__k": [90, 120, 180, "all"],
        "clf__n_estimators": [400, 700],
        "clf__max_depth": [None, 20, 35],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__max_features": ["sqrt", 0.35, 0.5],
    }
    return pipeline, params


if __name__ == "__main__":
    train_and_evaluate(NAME, get_search_config)
