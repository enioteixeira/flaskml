import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Gerando dados fictícios (100 amostras, 2 features)
np.random.seed(42)
X = np.random.rand(100, 2)  # Features
y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Labels: 0 ou 1

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinando o modelo
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Avaliando o modelo
y_pred = clf.predict(X_test)
print(f"Acurácia: {accuracy_score(y_test, y_pred):.2f}")

# Salvando o modelo em disco
with open("modelo.pkl", "wb") as file:
    pickle.dump(clf, file)

print("Modelo salvo como modelo.pkl")