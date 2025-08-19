import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report


df = pd.read_csv("19-08\\es1\\dataset\\rsc\\AirQualityUCI.csv", sep=';', decimal=',')
#rimuovo i valori nulli e le colonne vuote
df = df.dropna(axis=1, how='all')
df = df.replace(-200, np.nan)

df = df.dropna(subset=['CO(GT)'])

#converto in oggetti Datetime
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['DayOfWeek'] = df['Date'].dt.dayofweek

#definisco la variabile target confrontandola con la media giornaliera
df['CO_daily_mean'] = df.groupby(df['Date'])['CO(GT)'].transform('mean')
df['target'] = (df['CO(GT)'] > df['CO_daily_mean']).astype(int)

X = df[['CO(GT)', 'DayOfWeek']]
y = df['target']

#splitto in train e test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)
print("Decision Tree Results:\n")
print(classification_report(y_test, y_pred_dt))


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(256, 128,64, 32, 8), max_iter=500, random_state=42)
mlp.fit(X_train_scaled, y_train)

y_pred_mlp = mlp.predict(X_test_scaled)
print("MLP Results:\n")
print(classification_report(y_test, y_pred_mlp))
