import warnings

from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import RSS_FILL_VALUE
from train_model import train_and_evaluate

warnings.filterwarnings("ignore")

NAME = "KNN"


def get_search_config():
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=RSS_FILL_VALUE)),
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(score_func=mutual_info_classif)),
            ("clf", KNeighborsClassifier()),
        ]
    )
    params = {
        "selector__k": [60, 90, 120, 180],
        "clf__n_neighbors": [3, 5, 7, 9],
        "clf__weights": ["distance"],
        "clf__metric": ["manhattan", "euclidean"],
    }
    return pipeline, params


if __name__ == "__main__":
    train_and_evaluate(NAME, get_search_config)
