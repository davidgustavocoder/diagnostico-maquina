import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ======================
# Carregar modelo
# ======================

with open("modelo.pkl", "rb") as f:
    modelo = pickle.load(f)

le_ruido = LabelEncoder()
le_ruido.fit(["baixo", "medio", "alto"])

# ======================
# Interface
# ======================

st.title("🔧 Sistema de Diagnóstico de Máquina")

temperatura = st.number_input("Temperatura (°C)", value=50.0)
vibracao = st.number_input("Vibração (mm/s)", value=5.0)
tempo_operacao = st.number_input("Tempo de operação (horas)", value=100.0)

ruido = st.selectbox("Nível de ruído", ["baixo", "medio", "alto"])

if st.button("Analisar"):

    ruido_transformado = le_ruido.transform([ruido])[0]

    nova_entrada = [[temperatura, vibracao, ruido_transformado, tempo_operacao]]

    previsao = modelo.predict(nova_entrada)[0]
    probabilidades = modelo.predict_proba(nova_entrada)[0]
    classes = modelo.classes_

    st.subheader("Resultado")
    st.success(f"Falha provável: {previsao}")

    st.subheader("Probabilidades")

    for classe, prob in sorted(zip(classes, probabilidades), key=lambda x: x[1], reverse=True):
        st.write(f"{classe}: {prob*100:.2f}%")
