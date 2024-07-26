import pandas as pd


def main():
    df = pd.read_csv("../datasets/dataset_full.csv", low_memory=False)

    date_pattern = r'(\d{4}-\d{2}-\d{2})'

    df['date'] = df['_time'].str.extract(date_pattern)

    df['date'] = pd.to_datetime(df['date'])

    lista_datas = df['date'].dt.date.unique()

    for data in lista_datas:
        df_data = df[df['date'].dt.date == data]
        df_data.to_csv(
            f"../datasets/dataset_full_{data}.csv", index=False)


if __name__ == "__main__":
    main()
