import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random

#  dataset mentah Weather 
DATA_PATH = "weather_data.csv"
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print(df.head())

# seleksi x dan y 
feature_col = "Humidity_pct"
target_col = "Temperature_C"
df[feature_col] = pd.to_numeric(df[feature_col], errors="coerce")
df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

# untuk buang data kosong  
df_clean = df[[feature_col, target_col]].dropna()
print("\nData Setelah Dibersihkan:", df_clean.shape)

# visualisasi data (scatter) 
plt.figure(figsize=(10, 6))
plt.scatter(df_clean[feature_col], df_clean[target_col])
plt.xlabel("Humidity (%)")
plt.ylabel("Temperature (C)")
plt.title("Hubungan Humidity vs Temperature (Weather Dataset)")
plt.grid(True)
plt.show()

# untuk split data latih dan data uji 
X = df_clean[[feature_col]]
y = df_clean[target_col]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nJumlah Data Latih:", X_train.shape[0])
print("Jumlah Data Uji:", X_test.shape[0])

# model regresi linear 
model = LinearRegression()
model.fit(X_train, y_train)

# untuk prediksi data uji 
y_pred = model.predict(X_test)

# evaluasi model (R-squared) 
r_squared = model.score(X_test, y_test)
print("\nR-squared:", r_squared)

# untuk visualisasi hasil prediksi 
plt.figure(figsize=(10, 6))
plt.scatter(X_test[feature_col], y_test, label="Data Uji")
plt.scatter(X_test[feature_col], y_pred, label="Prediksi", marker="x")
plt.xlabel("Humidity (%)")
plt.ylabel("Temperature (C)")
plt.title("Prediksi Temperature berdasarkan Humidity (Weather Dataset)")
plt.legend(loc="upper right")
plt.grid(True)
plt.show()

# untuk memprediksi data baru secara acak  
min_h = int(df_clean[feature_col].min())
max_h = int(df_clean[feature_col].max())
random_h = random.randint(min_h, max_h)
new_humidity = pd.DataFrame({feature_col: [random_h]})
predicted_temp = model.predict(new_humidity)

print(f"\nPrediksi Temperature Untuk Humidity {random_h}:", predicted_temp[0])
