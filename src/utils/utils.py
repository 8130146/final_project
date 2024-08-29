import pickle
import pandas as pd


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


def datasetInfo():
    dataset = pd.read_csv("../datasets/dataset_short.csv")
    # Imprimir resumo básico
    print("Resumo do Dataset:")
    print(f"Total de linhas: {dataset.shape[0]}")
    print(f"Total de colunas: {dataset.shape[1]}")
    print("\nColunas e seus tipos de dados:")
    print(dataset.dtypes)
    print("\nEstatísticas descritivas das colunas numéricas:")
    print(dataset.describe())


def main():
    datasetInfo()


if __name__ == "__main__":
    main()
