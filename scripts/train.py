# scripts/train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

os.makedirs("../data", exist_ok=True)
os.makedirs("../outputs", exist_ok=True)

df = pd.read_csv("C:/Users/ADMIN/Desktop/model-monitoring-project/data.csv", encoding='latin1')
df.columns = df.columns.str.strip()
df = df.dropna(subset=['CustomerID'])
le_country = LabelEncoder()
df['Country_encoded'] = le_country.fit_transform(df['Country'])
df['TotalInvoiceValue'] = df['Quantity'] * df['UnitPrice']
df['Target'] = (df['TotalInvoiceValue'] > 100).astype(int)

X = df[['Quantity', 'UnitPrice', 'Country_encoded']]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "../outputs/model.pkl")
joblib.dump(le_country, "../outputs/le_country.pkl")

df[['Quantity', 'UnitPrice', 'Country_encoded', 'Target']].to_csv("../data/data_initial.csv", index=False)

print("Model trained and saved!")


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(accuracy)
