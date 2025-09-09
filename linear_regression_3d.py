import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# ipload data spreadshet
url = "https://docs.google.com/spreadsheets/d/1LfFRYtdg365-5vPddZ0c3kR43FxNdaroKk-w0dVimRk/export?format=csv&gid=1830137951"
df = pd.read_csv(url)

print("Preview dataset:")
print(df.head())

# fitur
X1 = df["Tinggi (cm)"].values
X2 = df["Lingkar Tubuh (cm)"].values
y = df["Berat (kg)"].values

# fitur --> matriks
X = np.column_stack((X1, X2))

# linReg sklearn
model = LinearRegression()
model.fit(X, y)

# koefisien , intercept
print("Intercept (b0):", model.intercept_)
print("Coef (b1, b2):", model.coef_)

# prediksi 
y_pred = model.predict(X)

# plot data + regresi
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# scatter data asli
ax.scatter(X1, X2, y, color='blue', label="Data Asli")

# grid regresi
x1_grid, x2_grid = np.meshgrid(
    np.linspace(X1.min(), X1.max(), 20),
    np.linspace(X2.min(), X2.max(), 20)
)
y_grid = model.intercept_ + model.coef_[0]*x1_grid + model.coef_[1]*x2_grid

# plot regresi
ax.plot_surface(x1_grid, x2_grid, y_grid, alpha=0.5, color='orange')

# label
ax.set_xlabel("Tinggi (cm)")
ax.set_ylabel("Lingkar Tubuh (cm)")
ax.set_zlabel("Berat (kg)")
ax.set_title("3D Linear Regression (Dataset Spreadsheet)")

plt.legend()
plt.show()