from utils import utils
import streamlit as st

import sys
sys.path.append('../utils')


def main():
    st.title('Decision Tree')
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
