import streamlit as st
import pandas as pd
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


def main():

    # Carregar dados salvos
    with open('metrics/data_naive_bayes.pkl', 'rb') as f:
        means, variances, feature_names, target_names = pickle.load(f)

    st.title("Naive Bayes")

    st.header("Metrics")
    st.markdown("##### Test Size: 0.2")
    st.markdown("##### Random State: 42")
    metrics_nb = load_metrics("./metrics/metrics_naive_bayes.pkl")
    st.markdown(f"##### Accuracy: {metrics_nb['Accuracy']}")
    st.markdown(f"##### Precision: {metrics_nb['Precision']}")
    st.markdown(f"##### Recall: {metrics_nb['Recall']}")
    st.markdown(f"##### F1: {metrics_nb['F1']}")

    st.header("Distribuicoes das caracteristicas por classe")
    # Select box for choosing the feature
    selected_feature = st.selectbox('Escolha a Característica', feature_names)
    st.subheader(f'Distribuição da Característica: {selected_feature}')
    st.image(f'images/distribuicao_{selected_feature}.png')

    st.header("Médias e Variâncias")
    st.subheader("Médias das Características por Classe")
    st.write(pd.DataFrame(means, index=target_names, columns=feature_names))
    st.subheader("Variâncias das Características por Classe")
    st.write(pd.DataFrame(variances, index=target_names, columns=feature_names))
