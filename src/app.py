import streamlit as st
import altair as alt
from paginas import page1, page2, page3

st.set_page_config(
    page_title="Analysis of the amari dataset",
    layout="centered",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")


pages = {
    "Analise Inicial do dataset": page1.main,
    "Decision Tree": page2.main,
    "Naive Bayes": page3.main
}

if __name__ == "__main__":
    st.sidebar.title("Menu")
    selection = st.sidebar.radio("Escolha uma página", list(pages.keys()))
    pages[selection]()
