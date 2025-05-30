import streamlit as st
from reveal_slides import slides

def app():
    st.header("Презентация проекта")

    # Настройки слайдов (стр. 16 методички)
    with st.sidebar:
        st.subheader("Настройки презентации")
        theme = st.selectbox("Тема", ["night", "solarized", "beige"])
        transition = st.selectbox("Переход", ["slide", "fade", "zoom"])

    # Контент слайдов
    slides_content = """
    ## Прогнозирование отказов оборудования
    ---
    ### Введение
    - **Задача**: Предсказать отказ оборудования (1) или его отсутствие (0).
    - **Датасет**: 10 000 записей, 14 признаков (температура, скорость вращения и др.).
    ---
    ### Этапы работы
    1. Загрузка данных (CSV или `st.file_uploader`).
    2. Предобработка:
        - Удаление лишних столбцов.
        - Кодирование категориальных признаков.
        - Масштабирование.
    3. Обучение моделей:
        - Logistic Regression.
        - Random Forest.
        - XGBoost.
    4. Оценка через Accuracy и ROC-AUC.
    ---
    ### Результаты
    - Лучшая модель: **XGBoost** (Accuracy=0.9850, AUC=0.9690).
    - Streamlit-интерфейс для предсказаний.
    """

    # Отображение слайдов
    slides(
        slides_content,
        height=500,
        theme=theme,
        config={"transition": transition}
    )