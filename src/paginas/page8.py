import streamlit as st
import pandas as pd
import os
import pickle
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt


def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model


def createSeriesDataset(df):
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

    return df


def preprocess_data(df, columns_to_check):
    # Drop linhas com valores NaN nas colunas especificadas
    df_cleaned = df.dropna(subset=columns_to_check)

    return df_cleaned


def generate_lime_explanation(model, X, instance_index):
    feature_names = X.columns.tolist()
    X = pd.DataFrame(X, columns=feature_names)
    explainer = lime.lime_tabular.LimeTabularExplainer(X.values, mode='classification', feature_names=feature_names, class_names=[
                                                       "Nao ataque", "Ataque DDoS"], verbose=True, random_state=42)
    instance = X.loc[instance_index]
    explanation = explainer.explain_instance(
        instance.values, model.predict_proba, num_features=len(feature_names))

    fig = explanation.as_pyplot_figure()
    st.pyplot(fig)
    plt.close(fig)


def main():
    st.title('Fazer previsao')

    st.write("Faça upload de um dataset(csv)")
    file = st.file_uploader("Selecione o dataset.", type=["csv"])

    if file is not None:
        df = pd.read_csv(file)
        st.write("Dataset carregado com sucesso!")
        st.write(df.head())

        st.write("Escolha um dos modelos treinados")
        models = [f for f in os.listdir("./models") if f.endswith('.pkl')]
        selected_model = st.selectbox('Selecione um modelo', models)

        if selected_model:
            model_path = os.path.join("./models", selected_model)
            model = load_model(model_path)
            st.write(f"Modelo {selected_model} carregado com sucesso!")

            columns = [
                'dl_bitrate', 'ul_bitrate', 'cell_x_dl_retx', 'cell_x_dl_tx',
                'cell_x_ul_retx', 'cell_x_ul_tx', 'ul_total_bytes_non_incr', 'dl_total_bytes_non_incr',
            ]

            # Preprocessar os dados
            df_preprocessed = preprocess_data(df, columns)

            if all(col in df_preprocessed.columns for col in columns):
                if selected_model.endswith('series.pkl'):
                    # transformar dataset em series
                    df_preprocessed = createSeriesDataset(df_preprocessed)
                    st.write("Dataset transformado em series!")
                    columns = [
                        'dl_bitrate', 'ul_bitrate', 'cell_x_dl_retx', 'cell_x_dl_tx',
                        'cell_x_ul_retx', 'cell_x_ul_tx', 'ul_total_bytes_non_incr', 'dl_total_bytes_non_incr',
                        'dl_bitrate_t-1', 'ul_bitrate_t-1', 'cell_x_dl_retx_t-1', 'cell_x_dl_tx_t-1',
                        'cell_x_ul_retx_t-1', 'cell_x_ul_tx_t-1', 'ul_total_bytes_non_incr_t-1', 'dl_total_bytes_non_incr_t-1',
                        'dl_bitrate_t-2', 'ul_bitrate_t-2', 'cell_x_dl_retx_t-2', 'cell_x_dl_tx_t-2',
                        'cell_x_ul_retx_t-2', 'cell_x_ul_tx_t-2', 'ul_total_bytes_non_incr_t-2', 'dl_total_bytes_non_incr_t-2',
                    ]
                    df_preprocessed = preprocess_data(df_preprocessed, columns)
                X = df_preprocessed[columns]
                predictions = model.predict(X)
                df_preprocessed['Predictions'] = predictions
                st.write("Previsões feitas com sucesso!")
                st.write(df_preprocessed)

                st.title("LIME Explanation for Model Predictions")
                # Utilizador seleciona uma instancia do dataset
                instance_index = st.selectbox(
                    "Select an instance for LIME explanation", X.index)
                if st.button("Generate LIME Plot"):
                    generate_lime_explanation(model, X, instance_index)
            else:
                st.write(
                    "O dataset não contém todas as colunas necessárias para fazer previsões.")


if __name__ == "__main__":
    main()
