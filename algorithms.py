import extra_trees
import knn
import random_forest
import svm

ALL_ALGORITHMS = {
    knn.NAME: knn.get_search_config,
    svm.NAME: svm.get_search_config,
    random_forest.NAME: random_forest.get_search_config,
    extra_trees.NAME: extra_trees.get_search_config,
}


def build_search_spaces():
    return {name: config() for name, config in ALL_ALGORITHMS.items()}
