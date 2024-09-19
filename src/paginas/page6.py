import streamlit as st
import pandas as pd


def main():
    st.title("Analise LIME para Decison Tree")

    # Carregar previsões das instâncias
    predictions_df = pd.read_csv(
        'images/lime_dTree/lime_dTree_predictions.csv')
    instance_predictions = dict(
        zip(predictions_df['instancia'], predictions_df['previsão']))

    st.header("Graficos LIME de explicacao para a instancia escolhida")

    selected_instance = st.selectbox(
        'Escolha a Instancia', list(instance_predictions.keys()))
    prediction = instance_predictions[selected_instance]
    st.subheader(f'Instancia: {selected_instance} | Previsao: {prediction}')
    st.image(f'images/lime_dTree/lime_dTree_instance_{selected_instance}.png')
    # Exibir arquivo CSV correspondente
    contributions_df = pd.read_csv(
        f'images/lime_dTree/lime_dTree_contributions_instance_{selected_instance}.csv')
    st.write(contributions_df)
