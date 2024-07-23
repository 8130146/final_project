import pandas as pd
import streamlit as st
import altair as alt
import os


@st.cache_data
def load_data(imei):
    filename = f"./datasets/dataset_full_imeisv_{imei}.csv"
    df = pd.read_csv(filename, low_memory=False)
    # Amostra dos dados
    # sample_size = 50000  # Tamanho da amostra
    # sampled_data = df.sample(min(sample_size, len(df)))
    return df


def main():
    st.title('Amari UE dataset Dashboard')

    datasets = os.listdir('./datasets')
    imeis_list = [file.split('_')[-1].split('.')[0]
                  for file in datasets if file.startswith('dataset_full_imeisv_')]
    selected_imei = st.selectbox('Select an IMEI', imeis_list)
    if selected_imei:
        df_selected_ue = load_data(selected_imei)
        st.write("Selected Imei: " + selected_imei)

    st.write(df_selected_ue)

    df_selected_ue['_time'] = pd.to_datetime(
        df_selected_ue['_time'], format='ISO8601')

    st.markdown('### Analise de trafego 5G')

    start_date, end_date = st.date_input('Select a Date range', [
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
    st.markdown("##### Taxa de bits uplink da celula associada ao dispositivo")
    bandwidth_chart = alt.Chart(df_selected_ue).mark_area().encode(
        x='_time:T',
        y='cell_x_ul_bitrate:Q'
    ).properties(
        width=800,
        height=400
    ).interactive()
    st.altair_chart(bandwidth_chart)
