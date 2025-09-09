import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as lnr
import numpy as np
import json

# api data curah hujan bmkg
kode_wilayah = "35.78.09.1001"  # ganti kode wilayah disini https://kodewilayah.id/
url = f"https://api.bmkg.go.id/publik/prakiraan-cuaca?adm4={kode_wilayah}"

response = requests.get(url)
print("Status code:", response.status_code)

if response.status_code != 200:
    raise Exception("❌ Gagal ambil data dari BMKG API")

data = response.json()

print("Top Level keys:", data.keys())

# data extraction: suhu, kelembapan, curah hujan
records = []
cuaca_data = data["data"][0]["cuaca"]  # list

for group in cuaca_data:    
    for d in group:           
        suhu = d.get("t")
        hum = d.get("hu")
        rain = d.get("tp", 0) 
        waktu = d.get("local_datetime")
        if suhu is not None and hum is not None:
            records.append([waktu, suhu, hum, rain])

# dataFrame
df = pd.DataFrame(records, columns=["datetime", "Temperature", "Humidity", "Rainfall"])
df["hour"] = pd.to_datetime(df["datetime"]).dt.hour

print("\n📊 Contoh dataset BMKG:\n", df.head())

# Linreg (prediksi suhu dari jam + kelembapan)
X = df[["hour", "Humidity"]]
y = df["Temperature"]

model = lnr()
model.fit(X, y)

# temp predict jam 0-23 dengan kelembapan rata-rata
X_pred = pd.DataFrame({
    "hour": np.arange(0, 24),
    "Humidity": [df["Humidity"].mean()] * 24
})
y_pred = model.predict(X_pred)

# plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(df["hour"], df["Humidity"], df["Temperature"], c="blue", label="Data Aktual")
ax.plot(X_pred["hour"], X_pred["Humidity"], y_pred, color="red", label="Linear Regression")

ax.set_xlabel("Jam (0-23)")
ax.set_ylabel("Kelembapan (%)")
ax.set_zlabel("Suhu (°C)")
ax.set_title("Prediksi Suhu berdasarkan Jam & Kelembapan (BMKG API)")
ax.legend()

plt.show()