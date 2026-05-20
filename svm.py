import warnings

from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import RANDOM_STATE, RSS_FILL_VALUE
from train_model import train_and_evaluate

warnings.filterwarnings("ignore")

NAME = "SVM"


def get_search_config():
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=RSS_FILL_VALUE)),
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(score_func=mutual_info_classif)),
            ("clf", SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE)),
        ]
    )
    params = {
        "selector__k": [90, 120, 180, "all"],
        "clf__C": [3, 10, 30],
        "clf__gamma": ["scale", 0.01, 0.03],
        "clf__kernel": ["rbf"],
    }
    return pipeline, params


if __name__ == "__main__":
    train_and_evaluate(NAME, get_search_config)
