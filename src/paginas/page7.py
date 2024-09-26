import streamlit as st
import pandas as pd


def main():
    st.title("LIME - Naive Bayes - dataset_short")

    # Carregar previsões das instâncias
    predictions_df = pd.read_csv(
        'images/lime_naiveBayes/lime_naiveBayes_predictions.csv')
    instance_predictions = dict(
        zip(predictions_df['instancia'], predictions_df['previsão']))

    st.header("Graficos LIME de explicacao para a instancia escolhida")

    selected_instance = st.selectbox(
        'Escolha a Instancia', list(instance_predictions.keys()))
    prediction = instance_predictions[selected_instance]
    st.subheader(f'Instancia: {selected_instance} | Previsao: {prediction}')
    st.image(
        f'images/lime_naiveBayes/lime_naiveBayes_instance_{selected_instance}.png')
    # Exibir arquivo CSV correspondente
    contributions_df = pd.read_csv(
        f'images/lime_naiveBayes/lime_naiveBayes_contributions_instance_{selected_instance}.csv')
    st.write(contributions_df)

    st.title("LIME - Naive Bayes - dataset_series")

    # Carregar previsões das instâncias
    predictions_df_series = pd.read_csv(
        'images/lime_naiveBayes_series/lime_naiveBayes_series_predictions.csv')
    instance_predictions_series = dict(
        zip(predictions_df_series['instancia'], predictions_df_series['previsão']))

    st.header("Graficos LIME de explicacao para a instancia escolhida (Series)")

    selected_instance_series = st.selectbox(
        'Escolha a Instancia (Series)', list(instance_predictions_series.keys()))
    prediction_series = instance_predictions_series[selected_instance_series]
    st.subheader(f'Instancia: {selected_instance_series} | Previsao: {
                 prediction_series}')
    st.image(
        f'images/lime_naiveBayes_series/lime_naiveBayes_series_instance_{selected_instance_series}.png')
    # Exibir arquivo CSV correspondente
    contributions_df_series = pd.read_csv(
        f'images/lime_naiveBayes_series/lime_naiveBayes_series_contributions_instance_{selected_instance_series}.csv')
    st.write(contributions_df_series)
