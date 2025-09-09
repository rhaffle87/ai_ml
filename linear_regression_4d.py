import pandas as pd
from sklearn.linear_model import LinearRegression

# 1. Load data dari Google Spreadsheet (CSV export link)
url = "https://docs.google.com/spreadsheets/d/1ZyRDLpLRoM-sKDFUxuUnZs-c5bGvRbTpX-XGyHS41sk/export?format=csv"
df = pd.read_csv(url)

# 2. Fitur = [Tinggi, Lingkar Tubuh, IMT], Target = Berat
X = df[["Tinggi (cm)", "Lingkar Tubuh (cm)", "IMT"]].values
y = df["Berat (kg)"].values

# 3. Multiple Linear Regression
model = LinearRegression()
model.fit(X, y)

# Koefisien & intercept
b0 = model.intercept_
b = model.coef_

print("Intercept (b0):", b0)
print("Koefisien (b):", b)

# Persamaan regresi
print(f"\nPersamaan regresi:")
print(f"y = {b0:.2f} + ({b[0]:.2f} * Tinggi) + ({b[1]:.2f} * Lingkar Tubuh) + ({b[2]:.2f} * IMT)")

# -----------------------------
# 4. Input manual dari user
# -----------------------------
try:
    tinggi = float(input("Masukkan Tinggi (cm): "))
    lingkar = float(input("Masukkan Lingkar Tubuh (cm): "))
    imt = float(input("Masukkan IMT: "))

    prediksi = model.predict([[tinggi, lingkar, imt]])
    print(f"\n[Manual] Prediksi berat badan untuk [Tinggi={tinggi}, Lingkar={lingkar}, IMT={imt}] = {prediksi[0]:.2f} kg")

except Exception as e:
    print("Error input:", e)

# -----------------------------
# 5. Prediksi batch (contoh 3 orang)
# -----------------------------
batch_data = [
    [165, 95, 24],   # orang 1
    [175, 100, 26],  # orang 2
    [160, 85, 22]    # orang 3
]

batch_pred = model.predict(batch_data)

print("\n[Batch] Hasil prediksi untuk beberapa orang:")
for data, pred in zip(batch_data, batch_pred):
    print(f"Tinggi={data[0]}, Lingkar={data[1]}, IMT={data[2]} → Prediksi Berat = {pred:.2f} kg")
