import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Load data dari Google Spreadsheet
url = "https://docs.google.com/spreadsheets/d/117mBNDNtrdYlfN6FmIg-3_WQ_lcuEsFWmSvpnMxR9fI/export?format=csv&gid=1164484126"
df = pd.read_csv(url)

# 2. Ambil fitur & target
X = df[["Tinggi (cm)"]].values
y = df["Berat (kg)"].values

# 3. Linear Regression
model = LinearRegression()
model.fit(X, y)

b0 = model.intercept_
b1 = model.coef_[0]

print(f"Persamaan regresi: y = {b0:.2f} + {b1:.2f} * x")

# 4. Prediksi untuk data asli
y_pred = model.predict(X)

# 5. Prediksi untuk tinggi tertentu
tinggi_tes = [[160], [165], [170], [175], [180]]
prediksi_bb = model.predict(tinggi_tes)

for t, p in zip(tinggi_tes, prediksi_bb):
    print(f"Tinggi {t[0]} cm → Prediksi Berat = {p:.2f} kg")

# 6. Visualisasi
plt.figure(figsize=(10,6))
plt.scatter(X, y, color="blue", label="Data Asli")
plt.plot(X, y_pred, color="red", linewidth=2, label=f"y = {b0:.2f} + {b1:.2f}x")

plt.scatter([t[0] for t in tinggi_tes], prediksi_bb, color="green", marker="x", s=100, label="Prediksi Titik Uji")

plt.xlabel("Tinggi (cm)")
plt.ylabel("Berat (kg)")
plt.title("Linear Regression: Tinggi vs Berat")
plt.legend()
plt.show()
