import streamlit as st
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


def main():
    st.title('Modelos Treinados e as suas métricas')
    st.markdown("### Decision Tree")
    st.markdown("##### Test Size: 0.2")
    st.markdown("##### Random State: 42")
    metrics_dt = load_metrics("./metrics/metrics_decision_tree.pkl")
    st.markdown(f"##### Accuracy: {metrics_dt['Accuracy']}")
    st.markdown(f"##### Precision: {metrics_dt['Precision']}")
    st.markdown(f"##### Recall: {metrics_dt['Recall']}")
    st.markdown(f"##### F1: {metrics_dt['F1']}")

    st.markdown("### Naive Bayes")
    st.markdown("##### Test Size: 0.2")
    st.markdown("##### Random State: 42")
    metrics_nb = load_metrics("./metrics/metrics_naive_bayes.pkl")
    st.markdown(f"##### Accuracy: {metrics_nb['Accuracy']}")
    st.markdown(f"##### Precision: {metrics_nb['Precision']}")
    st.markdown(f"##### Recall: {metrics_nb['Recall']}")
    st.markdown(f"##### F1: {metrics_nb['F1']}")
