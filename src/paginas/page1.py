import pandas as pd
import streamlit as st
import altair as alt
import os


@st.cache_data
def load_separate_by_imei(file):
    if file is not None:
        df = pd.read_csv(file)
        imeis_list = df['imeisv'].unique()
        imeis_dataframes = {}
        for imei in imeis_list:
            imeis_dataframes[imei] = df[df['imeisv'] == imei]
        return imeis_dataframes
    else:
        return {}


def main():
    st.title('Carregar dataset')
    st.write("Faça upload de um dataset(csv)")

    df = st.file_uploader("Selecione o dataset.", type=["csv"])
    imeis_dataframes = load_separate_by_imei(df)

    if imeis_dataframes:
        st.write("Dataset carregado com sucesso!")

        imeis_list = list(imeis_dataframes.keys())
        selected_imei = st.selectbox('Select an IMEI', imeis_list)

        if selected_imei:
            df_selected_ue = imeis_dataframes[selected_imei]
            st.write("Selected Imei: " + str(selected_imei))

        st.write(df_selected_ue)

        df_selected_ue['_time'] = pd.to_datetime(
            df_selected_ue['_time'], format='ISO8601')

        st.markdown('### Analise de trafego 5G')

        start_date, end_date = st.date_input('Selecione um intervalo de datas', [
            df_selected_ue['_time'].min().date(), df_selected_ue['_time'].max().date()])
        df_selected_ue = df_selected_ue[(df_selected_ue['_time'].dt.date >= start_date) & (
            df_selected_ue['_time'].dt.date <= end_date)]

        # Grafico para mostrar a taxa de bits ao Longo do Tempo
        st.markdown('##### Taxa de bits de downlink ao Longo do Tempo')
        chart_bits = alt.Chart(df_selected_ue).mark_line().encode(
            x='_time:T',
            y='dl_bitrate:Q',
        ).properties(
            width=800,
            height=400
        ).interactive()
        st.altair_chart(chart_bits)

        # Gráfico de Linhas para Taxa de Bits de Uplink ao Longo do Tempo
        st.markdown('##### Taxa de Bits de Uplink ao Longo do Tempo')
        ul_bitrate_chart = alt.Chart(df_selected_ue).mark_line().encode(
            x='_time:T',
            y='ul_bitrate:Q'
        ).properties(
            width=800,
            height=400
        ).interactive()
        st.altair_chart(ul_bitrate_chart)

        # Gráfico de Linhas para Taxa de Retransmissões ao Longo do Tempo
        st.markdown(
            '##### Taxa de Retransmissões downlink da celula do dispositivo ao Longo do Tempo')
        chart_retransmissao = alt.Chart(df_selected_ue).mark_line().encode(
            x='_time:T',
            y='cell_x_dl_retx:Q',
        ).properties(
            width=800,
            height=400
        ).interactive()
        st.altair_chart(chart_retransmissao)

        # Gráfico de Área para Utilização da Banda por Período de Tempo
        st.markdown(
            "##### Taxa de bits uplink da celula associada ao dispositivo")
        bandwidth_chart = alt.Chart(df_selected_ue).mark_area().encode(
            x='_time:T',
            y='cell_x_ul_bitrate:Q'
        ).properties(
            width=800,
            height=400
        ).interactive()
        st.altair_chart(bandwidth_chart)
