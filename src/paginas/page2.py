from utils import utils
import streamlit as st

import sys
sys.path.append('../utils')


def main():
    st.title('Decision Tree treinado com dataset_short')
    st.markdown("### Metricas")
    st.markdown("##### Test Size: 0.2")
    st.markdown("##### Random State: 42")
    metrics_dt = utils.load_metrics("./metrics/metrics_decision_tree.pkl")
    st.markdown(f"##### Accuracy: {metrics_dt['Accuracy']}")
    st.markdown(f"##### Precision: {metrics_dt['Precision']}")
    st.markdown(f"##### Recall: {metrics_dt['Recall']}")
    st.markdown(f"##### F1: {metrics_dt['F1']}")

    st.markdown("### Grafico da Decision Tree")
    image_path = "./images/grafico_decision_tree.png"
    # Inicializa a variável de estado se ela não existir
    if 'mostrar_tamanho_real' not in st.session_state:
        st.session_state['mostrar_tamanho_real'] = False

    # Função para alternar o estado
    def toggle_image_size():
        st.session_state['mostrar_tamanho_real'] = not st.session_state['mostrar_tamanho_real']

    # Botão para alternar o tamanho da imagem
    if st.button('Aumentar ou Diminuir imagem'):
        toggle_image_size()

    # Verifica o estado atual e mostra a imagem de acordo
    if st.session_state['mostrar_tamanho_real']:
        st.image(image_path, caption="Imagem Carregada", width=7000)
    else:
        st.image(image_path, caption="Imagem Carregada", use_column_width=True)

    st.title('Decision Tree treinado com dataset_series')
    st.markdown("### Metricas")
    st.markdown("##### Test Size: 0.2")
    st.markdown("##### Random State: 42")
    metrics_dt_series = utils.load_metrics(
        "./metrics/metrics_decision_tree_series.pkl")
    st.markdown(f"##### Accuracy: {metrics_dt_series['Accuracy']}")
    st.markdown(f"##### Precision: {metrics_dt_series['Precision']}")
    st.markdown(f"##### Recall: {metrics_dt_series['Recall']}")
    st.markdown(f"##### F1: {metrics_dt_series['F1']}")

    st.markdown("### Grafico da Decision Tree")
    image_path_series = "./images/grafico_decision_tree_series.png"
    if 'mostrar_tamanho_real_series' not in st.session_state:
        st.session_state['mostrar_tamanho_real_series'] = False

    def toggle_image_size_series():
        st.session_state['mostrar_tamanho_real_series'] = not st.session_state['mostrar_tamanho_real_series']

    if st.button('Aumentar ou Diminuir imagem (Series)'):
        toggle_image_size_series()

    if st.session_state['mostrar_tamanho_real_series']:
        st.image(image_path_series, caption="Imagem Carregada", width=7000)
    else:
        st.image(image_path_series, caption="Imagem Carregada",
                 use_column_width=True)
