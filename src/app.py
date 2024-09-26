import streamlit as st
import altair as alt
from paginas import page1, page2, page3, page4, page5, page6, page7, page8

st.set_page_config(
    page_title="Deteção de ataques DDoS",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")


pages = {
    "Analise Inicial do dataset": page1.main,
    "Decision Tree": page2.main,
    "Naive Bayes": page3.main,
    "Analise SHAP Decicion Tree": page4.main,
    "Analise SHAP Naive Bayes": page5.main,
    "Analise LIME Decicion Tree": page6.main,
    "Analise LIME Naive Bayes": page7.main,
    "Fazer previsao": page8.main
}

if __name__ == "__main__":
    st.sidebar.title("Menu")
    selection = st.sidebar.radio("Escolha uma página", list(pages.keys()))
    pages[selection]()
