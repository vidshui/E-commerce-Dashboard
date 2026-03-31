import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score
import datetime
import os
os.makedirs("../outputs", exist_ok=True)

df = pd.read_csv("C:/Users/ADMIN/Desktop/model-monitoring-project/data.csv", encoding='latin1')
df.columns = df.columns.str.strip()

df = df.dropna(subset=['CustomerID'])

df['TotalInvoiceValue'] = df['Quantity'] * df['UnitPrice']
df['Target'] = (df['TotalInvoiceValue'] > 100).astype(int)

# Encode Country using encoder from train.py
le_country = joblib.load("../outputs/le_country.pkl")
df['Country_encoded'] = le_country.transform(df['Country'])

# Copy for monitoring simulation
df_old = df.copy()
df_new = df_old.copy()

# Simulate drift
df_new['Quantity'] += np.random.randint(-2, 3, size=df_new.shape[0])
df_new['UnitPrice'] *= np.random.uniform(0.95, 1.05, size=df_new.shape[0])

# Load model
model = joblib.load("../outputs/model.pkl")

# Features and target
X_new = df_new[['Quantity', 'UnitPrice', 'Country_encoded']]
y_true = df_new['Target']

# Predict
y_pred = model.predict(X_new)
accuracy = accuracy_score(y_true, y_pred)

# Calculate drift
drift_scores = [abs(df_old[col].mean() - df_new[col].mean()) for col in ['Quantity', 'UnitPrice', 'Country_encoded']]
avg_drift = np.mean(drift_scores)

# Log monitoring results
log_file = "../outputs/monitoring_log.csv"
if not os.path.isfile(log_file):
    pd.DataFrame(columns=['timestamp', 'accuracy', 'avg_drift']).to_csv(log_file, index=False)

pd.DataFrame([{
    'timestamp': datetime.datetime.now(),
    'accuracy': accuracy,
    'avg_drift': avg_drift
}]).to_csv(log_file, mode='a', header=False, index=False)

print("Monitoring run complete!")
print("Accuracy:", accuracy)
print("Avg Drift:", avg_drift)
