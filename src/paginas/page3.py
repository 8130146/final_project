import streamlit as st


def main():
    st.title("Graficos LIME")
    st.markdown("### Grafico LIME da Decision Tree")
    image_path = "./images/lime_decisionTree.png"
    st.image(image_path, caption="Imagem Carregada", use_column_width=True)

    st.markdown("### Grafico LIME do Naive Bayes")
    image_path = "./images/lime_naiveBayes.png"
    st.image(image_path, caption="Imagem Carregada", use_column_width=True)
