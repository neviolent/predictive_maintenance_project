import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, accuracy_score

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    st.warning("XGBoost не установлен. Сравнение будет выполнено только для Logistic Regression и Random Forest.")

def clean_feature_names(df):
    """Удаляет специальные символы из названий столбцов"""
    return df.rename(columns=lambda x: x.replace('[', '').replace(']', '').replace('<', ''))

def app():
    st.header("📊 Анализ данных и модель")

    # Загрузка данных
    uploaded_file = st.file_uploader("Загрузите CSV-файл", type="csv")
    if not uploaded_file:
        st.warning("Загрузите файл для продолжения.")
        return

    data = pd.read_csv(uploaded_file)
    
    # Очистка названий столбцов
    data = clean_feature_names(data)
    
    # Проверка пропущенных значений
    st.subheader("Проверка пропущенных значений")
    st.write(data.isnull().sum())

    # Визуализация распределений
    st.subheader("Распределение признаков")
    numerical_cols = [col for col in data.columns if data[col].dtype in ['float64', 'int64']]
    for col in numerical_cols:
        fig, ax = plt.subplots()
        sns.histplot(data[col], kde=True, ax=ax)
        st.pyplot(fig)

    # Предобработка
    drop_cols = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    data = data.drop(columns=[col for col in drop_cols if col in data.columns])
    if 'Type' in data.columns:
        data['Type'] = LabelEncoder().fit_transform(data['Type'])

    # Масштабирование
    scaler = StandardScaler()
    numerical_features = [col for col in data.columns if col not in ['Machine failure', 'Type']]
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    # Демонстрация предобработки
    st.subheader("Предобработанные данные (первые 5 строк)")
    st.dataframe(data.head())

    # Разделение данных
    X = data.drop(columns=['Machine failure'])
    y = data['Machine failure']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучение и сравнение моделей
    st.subheader("Сравнение моделей")
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(n_estimators=100, random_state=42)

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        results.append({"Модель": name, "Accuracy": accuracy, "ROC-AUC": roc_auc})

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        st.subheader(f"Confusion Matrix: {name}")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # Classification Report
        report = classification_report(y_test, y_pred, output_dict=True)
        st.subheader(f"Classification Report: {name}")
        st.dataframe(pd.DataFrame(report).transpose())

    st.table(pd.DataFrame(results))

    # ROC-кривые для всех моделей
    st.subheader("ROC-кривые")
    fig, ax = plt.subplots()
    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, y_pred_proba):.2f})")
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

    # Интерфейс для предсказаний
    st.subheader("Ручной ввод данных")
    with st.form("prediction_form"):
        st.write("Введите параметры оборудования:")
        type_ = st.selectbox("Тип оборудования (L=0, M=1, H=2)", [0, 1, 2])
        air_temp = st.number_input("Температура воздуха K", value=300.0)
        process_temp = st.number_input("Температура процесса K", value=310.0)
        rotational_speed = st.number_input("Скорость вращения rpm", value=1500)
        torque = st.number_input("Крутящий момент Nm", value=40.0)
        tool_wear = st.number_input("Износ инструмента min", value=0)
        
        if st.form_submit_button("Предсказать"):
            input_data = pd.DataFrame({
                "Type": [type_],
                "Air temperature K": [air_temp],
                "Process temperature K": [process_temp],
                "Rotational speed rpm": [rotational_speed],
                "Torque Nm": [torque],
                "Tool wear min": [tool_wear]
            })
            input_data[numerical_features] = scaler.transform(input_data[numerical_features])
            prediction = models["Random Forest"].predict(input_data)[0]
            st.success(f"Прогноз: {'Отказ' if prediction == 1 else 'Нет отказа'}")

if __name__ == "__main__":
    app()
