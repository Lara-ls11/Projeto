from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_model(name, model, x, y, encoder, dataset_name):
    pred_encoded = model.predict(x)
    pred = encoder.inverse_transform(pred_encoded)
    acc = accuracy_score(y, pred)
    print(f"\n{'-' * 60}")
    print(f"Modelo   : {name}")
    print(f"Conjunto : {dataset_name}")
    print(f"Accuracy : {acc:.4f} ({acc * 100:.2f}%)")
    print(confusion_matrix(y, pred))
    print(classification_report(y, pred))
    return acc
