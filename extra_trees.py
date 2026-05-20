import warnings

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from common import RANDOM_STATE, RSS_FILL_VALUE
from train_model import train_and_evaluate

warnings.filterwarnings("ignore")

NAME = "Extra Trees"


def get_search_config():
    pipeline = Pipeline(
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
    )
    params = {
        "selector__k": [90, 120, 180, "all"],
        "clf__n_estimators": [500, 800],
        "clf__max_depth": [None, 25, 40],
        "clf__min_samples_leaf": [1, 2],
        "clf__max_features": ["sqrt", 0.35, 0.5],
    }
    return pipeline, params


if __name__ == "__main__":
    train_and_evaluate(NAME, get_search_config)
