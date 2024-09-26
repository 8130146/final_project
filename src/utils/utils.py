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


def divideDtaatsetByDay():
    df = pd.read_csv("../datasets/dataset_full.csv", low_memory=False)

    date_pattern = r'(\d{4}-\d{2}-\d{2})'

    df['date'] = df['_time'].str.extract(date_pattern)

    df['date'] = pd.to_datetime(df['date'])

    lista_datas = df['date'].dt.date.unique()

    for data in lista_datas:
        df_data = df[df['date'].dt.date == data]
        df_data.to_csv(
            f"../datasets/dataset_full_{data}.csv", index=False)


def createSeriesDataset():
    df = pd.read_csv("../datasets/dataset_short.csv", low_memory=False)
    # Criar colunas deslocadas
    df['dl_bitrate_t-1'] = df['dl_bitrate'].shift(1)
    df['ul_bitrate_t-1'] = df['ul_bitrate'].shift(1)
    df['cell_x_dl_retx_t-1'] = df['cell_x_dl_retx'].shift(1)
    df['cell_x_dl_tx_t-1'] = df['cell_x_dl_tx'].shift(1)
    df['cell_x_ul_retx_t-1'] = df['cell_x_ul_retx'].shift(1)
    df['cell_x_ul_tx_t-1'] = df['cell_x_ul_tx'].shift(1)
    df['ul_total_bytes_non_incr_t-1'] = df['ul_total_bytes_non_incr'].shift(1)
    df['dl_total_bytes_non_incr_t-1'] = df['dl_total_bytes_non_incr'].shift(1)

    df['dl_bitrate_t-2'] = df['dl_bitrate'].shift(2)
    df['ul_bitrate_t-2'] = df['ul_bitrate'].shift(2)
    df['cell_x_dl_retx_t-2'] = df['cell_x_dl_retx'].shift(2)
    df['cell_x_dl_tx_t-2'] = df['cell_x_dl_tx'].shift(2)
    df['cell_x_ul_retx_t-2'] = df['cell_x_ul_retx'].shift(2)
    df['cell_x_ul_tx_t-2'] = df['cell_x_ul_tx'].shift(2)
    df['ul_total_bytes_non_incr_t-2'] = df['ul_total_bytes_non_incr'].shift(2)
    df['dl_total_bytes_non_incr_t-2'] = df['dl_total_bytes_non_incr'].shift(2)

    # Remover linhas com valores NaN resultantes da deslocacao
    df = df.dropna()
    df.to_csv("../datasets/dataset_series.csv", index=False)


def main():
    createSeriesDataset()


if __name__ == "__main__":
    main()
