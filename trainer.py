import pandas as pd 
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv(r"C:\Users\david\OneDrive\Desktop\dataset.csv", sep=";")
print("Dataset carregado:")
print(df.head())

le_ruido = LabelEncoder()
df["ruido"] = le_ruido.fit_transform(df["ruido"])

X = df[["temperatura", "vibracao", "ruido", "tempo_operacao"]]
y = df["falha"]

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42
)

modelo = RandomForestClassifier(n_estimators=100, random_state=42)

modelo.fit(X_treino, y_treino)

previsoes = modelo.predict(X_teste)

print("\nRelatório de desempenho:")
print(classification_report(y_teste, previsoes))

with open("modelo.pkl", "wb") as arquivo:
    pickle.dump(modelo, arquivo)

print("\nModelo salvo com sucesso como 'modelo.pkl'")