"""
Testa cada algoritmo em sequência e mostra um resumo final.

Uso:
    python testar_modelos.py
    python testar_modelos.py knn svm          # só os indicados
"""

import sys

import extra_trees
import knn
import random_forest
import svm
from train_model import train_and_evaluate

ALL_ALGORITHMS = {
    "knn": knn,
    "svm": svm,
    "random_forest": random_forest,
    "extra_trees": extra_trees,
}


def main():
    if len(sys.argv) > 1:
        keys = [arg.lower().replace("-", "_") for arg in sys.argv[1:]]
        unknown = [k for k in keys if k not in ALL_ALGORITHMS]
        if unknown:
            print(f"Algoritmos desconhecidos: {', '.join(unknown)}")
            print(f"Disponíveis: {', '.join(ALL_ALGORITHMS)}")
            sys.exit(1)
        to_run = [ALL_ALGORITHMS[k] for k in keys]
    else:
        to_run = list(ALL_ALGORITHMS.values())

    results = []
    for module in to_run:
        print("\n" + "#" * 60)
        print(f"# {module.NAME}")
        print("#" * 60)
        outcome = train_and_evaluate(module.NAME, module.get_search_config)
        results.append(
            {
                "name": module.NAME,
                "validation": outcome["validation_accuracy"],
                "test": outcome["test_accuracy"],
            }
        )

    print("\n" + "=" * 60)
    print("RESUMO")
    print("=" * 60)
    print(f"{'Modelo':<16} {'Validação':>12} {'Teste':>12}")
    print("-" * 42)
    for row in sorted(results, key=lambda r: r["test"], reverse=True):
        print(
            f"{row['name']:<16} "
            f"{row['validation'] * 100:>11.2f}% "
            f"{row['test'] * 100:>11.2f}%"
        )

    best = max(results, key=lambda r: r["test"])
    print("-" * 42)
    print(f"Melhor no teste: {best['name']} ({best['test'] * 100:.2f}%)")


if __name__ == "__main__":
    main()
