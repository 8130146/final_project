import streamlit as st
import altair as alt
from paginas import page1, page2, page3, page4, page5

st.set_page_config(
    page_title="Analysis of the amari dataset",
    layout="centered",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")


pages = {
    "Analise Inicial do dataset": page1.main,
    "Modelos treinados e as suas métricas": page2.main,
    "Graficos LIME": page3.main,
    "Graficos SHAP": page4.main,
    "Graficos dos Modelos": page5.main
}

if __name__ == "__main__":
    st.sidebar.title("Menu")
    selection = st.sidebar.radio("Escolha uma página", list(pages.keys()))
    pages[selection]()
