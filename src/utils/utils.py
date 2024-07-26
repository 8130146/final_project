import pickle


def load_metrics(metrics_path):

    with open(metrics_path, 'rb') as file:
        metrics = pickle.load(file)
    metrics = {
        'Accuracy': metrics['accuracy'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1': metrics['f1']
    }
    return metrics


def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model
