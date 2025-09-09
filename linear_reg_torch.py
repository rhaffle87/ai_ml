import model_train
import matplotlib.pyplot as plt
import requests
import io
import csv

model_train.set_printoptions(precision=3)

# Ambil data dari Google Spreadsheet 
url = "https://docs.google.com/spreadsheets/d/1GZpimRB5VEW5Ea3vrxhKcpLQHO5RkGXnH0-DTnysJ5Q/export?format=csv&gid=0"
response = requests.get(url)
content = response.content.decode("utf-8")

reader = csv.reader(io.StringIO(content))
header = next(reader)  
rows = list(reader)

# Konversi ke tensor
x1 = model_train.tensor([float(r[0]) for r in rows], dtype=model_train.float32)  # Umur
x2 = model_train.tensor([float(r[1]) for r in rows], dtype=model_train.float32)  # Kecepatan
y  = model_train.tensor([float(r[2]) for r in rows], dtype=model_train.float32)  # Jarak

print(f'x1 = {x1}')
print(f'x2 = {x2}')
print(f'y  = {y}')

# Bentuk matriks desain (X) 
X_train = model_train.cat(tensors=(
    model_train.ones_like(x1).unsqueeze(dim=1),  # kolom bias
    x1.unsqueeze(dim=1),
    x2.unsqueeze(dim=1)
), dim=1)

y_train = y.unsqueeze(dim=1)

print(f'X_train = {X_train}')
print(f'y_train = {y_train}')

# Normal equation: w = (XᵀX)^-1 Xᵀy
w_vect = model_train.inverse(X_train.T @ X_train) @ X_train.T @ y_train

w0_pred = w_vect[0].item()
w1_pred = w_vect[1].item()
w2_pred = w_vect[2].item()

print(f'w0_pred = {w0_pred:.3f} | w1_pred = {w1_pred:.3f} | w2_pred = {w2_pred:.3f}')

# Prediksi
y_pred = (X_train @ w_vect).squeeze()

# Visualisasi 3D 
fig = plt.figure()
ax_3D = fig.add_subplot(projection='3d')
ax_3D.scatter(x1, x2, y, color="blue", label="Data Asli")

# Grid permukaan regresi
X1, X2 = model_train.meshgrid(
    model_train.linspace(x1.min(), x1.max(), 20),
    model_train.linspace(x2.min(), x2.max(), 20),
    indexing="ij"
)
Y = w0_pred + w1_pred * X1 + w2_pred * X2

ax_3D.plot_surface(X1, X2, Y, color="orange", alpha=0.5)

ax_3D.set_xlabel("Umur")
ax_3D.set_ylabel("Kecepatan (Km/jam)")
ax_3D.set_zlabel("Jarak (Km)")
ax_3D.set_title("Regresi linear data pelari")

plt.legend()
plt.show()
