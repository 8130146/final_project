import pandas as pd


def main():
    df = pd.read_csv("../datasets/dataset_full.csv", low_memory=False)
    ue_list = df['imeisv'].unique()

    for imei in ue_list:
        df_imei = df[df['imeisv'] == imei]
        df_imei.to_csv(
            f"../datasets/dataset_full_imeisv_{imei}.csv", index=False)


if __name__ == "__main__":
    main()
