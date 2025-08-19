import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("19-08\\es1\\dataset\\rsc\\energy_dataset.csv")
df["time"] = pd.to_datetime(df["time"], utc=True)
#print(df.columns.to_list())

df["hour"] = df["time"].dt.hour
df["dayofweek"] = df["time"].dt.dayofweek
df["month"] = df["time"].dt.month

media_per_ora = df.groupby('hour')['total load actual'].mean()

df['media_ora_specifica'] = df['hour'].map(media_per_ora)

df['target'] = (df['total load actual'] > df['media_ora_specifica']).astype(int)

X_temporal = df[["hour", "dayofweek", "month"]]

X = X_temporal
y = df["target"]

print(f"\nShape dataset: X={X.shape}, y={y.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.25, random_state=42
)
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)

print("Decision Tree:")
print(classification_report(y_test, y_pred_tree, digits=3))