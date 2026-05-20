import pandas as pd

from common import prepare_data
from evaluate import evaluate_model
from selection import run_single_model_selection


def train_and_evaluate(name, get_search_config):
    train_x, train_y, val_x, val_y, test_x, test_y_raw, encoder = prepare_data()
    pipeline, params = get_search_config()

    best_models, validation_scores = run_single_model_selection(
        train_x, train_y, val_x, val_y, name, pipeline, params
    )
    model = best_models[name]
    print(f"\nMelhor accuracy de validação: {validation_scores[name]:.4f}")

    full_train_x = pd.concat([train_x, val_x], ignore_index=True)
    full_train_y = pd.concat([train_y, val_y], ignore_index=True)
    model.fit(full_train_x, full_train_y)

    print("\n" + "=" * 60)
    print(f"RESULTADOS — {name}")
    print("=" * 60)
    test_acc = evaluate_model(name, model, test_x, test_y_raw, encoder, "Teste")
    return {
        "model": model,
        "validation_accuracy": validation_scores[name],
        "test_accuracy": test_acc,
    }
