import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ==============================
# 1️⃣ Carregar modelo treinado
# ==============================

with open("modelo.pkl", "rb") as f:
    modelo = pickle.load(f)

# ==============================
# 2️⃣ Recriar encoder do ruído
# ==============================

le_ruido = LabelEncoder()
le_ruido.fit(["baixo", "medio", "alto"])

print("=== SISTEMA DE DIAGNÓSTICO DE MÁQUINA ===")

while True:

    try:
        # ==============================
        # 3️⃣ Entrada do usuário
        # ==============================

        temperatura = float(input("\nDigite a temperatura (°C): "))
        vibracao = float(input("Digite a vibração (mm/s): "))
        ruido = input("Digite o nível de ruído (baixo, medio, alto): ").lower()
        tempo_operacao = float(input("Digite o tempo de operação (horas): "))

        if ruido not in ["baixo", "medio", "alto"]:
            print("⚠ Nível de ruído inválido. Tente novamente.")
            continue

        ruido_transformado = le_ruido.transform([ruido])[0]

        nova_entrada = [[temperatura, vibracao, ruido_transformado, tempo_operacao]]

        # ==============================
        # 4️⃣ Previsão
        # ==============================

        previsao = modelo.predict(nova_entrada)[0]
        probabilidades = modelo.predict_proba(nova_entrada)[0]
        classes = modelo.classes_

        print("\n===== RESULTADO DA IA =====")
        print(f"🔧 Falha provável: {previsao}")

        print("\n📊 Probabilidades:")
        for classe, prob in sorted(zip(classes, probabilidades), key=lambda x: x[1], reverse=True):
            print(f"{classe}: {prob*100:.2f}%")

        # ==============================
        # 5️⃣ Perguntar se deseja continuar
        # ==============================

        continuar = input("\nDeseja fazer outra análise? (s/n): ").lower()
        if continuar != "s":
            print("Encerrando sistema...")
            break

    except ValueError:
        print("⚠ Entrada inválida. Digite apenas números onde solicitado.")
