import warnings

import pandas as pd

from common import prepare_data
from ensemble import build_weighted_ensemble
from evaluate import evaluate_model
from algorithms import build_search_spaces
from selection import run_model_selection

warnings.filterwarnings("ignore")


def main():
    train_x, train_y, val_x, val_y, test_x, test_y_raw, encoder = prepare_data()
    best_models, validation_scores = run_model_selection(
        train_x, train_y, val_x, val_y, build_search_spaces()
    )

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
