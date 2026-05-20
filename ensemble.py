from sklearn.ensemble import VotingClassifier


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
