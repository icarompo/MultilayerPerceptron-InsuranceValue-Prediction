import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

df = pd.read_csv("./insurance.csv")

X = df.drop("charges", axis=1)
y = df["charges"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#tratamento
categorical_columns = ['sex', 'smoker', 'region']
X_train = pd.get_dummies(X_train, columns=categorical_columns, drop_first=True)
X_test = pd.get_dummies(X_test, columns=categorical_columns, drop_first=True)

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

#parâmetros
model = MLPRegressor(random_state=42, max_iter=10000, hidden_layer_sizes=(300, 150, 75), learning_rate='adaptive', learning_rate_init=0.001)

#normalização
pipeline = Pipeline(steps=[
    ('scaler', numeric_transformer),
    ('model', model)
])

#treinamento
pipeline.fit(X_train, y_train)

#previsoes
y_pred = pipeline.predict(X_test)

#erro medio absoluto e erro medio quadrado
mae = round(mean_absolute_error(y_test, y_pred), 2)
print(f"Mean Absolute Error: {mae}")

mse = round(mean_squared_error(y_test, y_pred), 2)
print(f"Mean Squared Error: {mse}")



