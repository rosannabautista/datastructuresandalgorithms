# main.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Cargar datos simulados (puedes cambiarlo por un dataset real)
# Ejemplo: historial de preferencias de los usuarios
# Cada fila representa un usuario con diferentes características
data = {
    'age': [25, 32, 47, 51, 62, 19],
    'gender': [0, 1, 0, 1, 0, 1],  # 0 = Masculino, 1 = Femenino
    'previous_purchase': [0, 1, 1, 0, 1, 0],  # 0 = No, 1 = Sí
    'likes_tech': [1, 0, 1, 0, 1, 1],  # 1 = Sí, 0 = No
    'recommended_product': [1, 0, 1, 0, 1, 0]  # 1 = Tech Product, 0 = Otro
}

# Convertir datos en un DataFrame de pandas
df = pd.DataFrame(data)

# Seleccionar características (features) y etiquetas (labels)
X = df[['age', 'gender', 'previous_purchase', 'likes_tech']]  # Features
y = df['recommended_product']  # Labels (producto recomendado)

# Dividir los datos en conjunto de entrenamiento y pruebas
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear un modelo de árbol de decisión
model = DecisionTreeClassifier()

# Entrenar el modelo
model.fit(X_train, y_train)

# Hacer predicciones con el conjunto de pruebas
y_pred = model.predict(X_test)

# Evaluar la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy * 100:.2f}%')

# Ejemplo de recomendación para un nuevo usuario
new_user = [[30, 1, 0, 1]]  # Edad 30, Femenino, No compró antes, Le gusta la tecnología
predicted = model.predict(new_user)

if predicted == 1:
    print("Recomendación: Producto tecnológico")
else:
    print("Recomendación: Otro producto")
