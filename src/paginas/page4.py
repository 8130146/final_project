import streamlit as st


def main():
    st.title("Graficos SHAP")
    st.markdown("### Grafico SHAP da Decision Tree")
    image_path = "./images/shap_decision_tree.png"
    st.image(image_path, caption="Imagem Carregada", use_column_width=True)

    st.markdown("### Grafico SHAP do Naive Bayes")
    image_path = "./images/shap_naive_bayes.png"
    st.image(image_path, caption="Imagem Carregada", use_column_width=True)
