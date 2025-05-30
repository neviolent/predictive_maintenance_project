import streamlit as st
import analysis_and_model
import presentation

# Настройка навигации (стр. 17 методички)
st.set_page_config(page_title="Predictive Maintenance", layout="wide")
st.sidebar.title("Навигация")
page = st.sidebar.radio("Выберите страницу", ["Анализ и модель", "Презентация"])

if page == "Анализ и модель":
    analysis_and_model.app()
else:
    presentation.app()