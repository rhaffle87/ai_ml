import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Data awal
x = [4, 5, 10, 4, 3, 11, 14 , 8, 10, 12, 31, 53, 23, 45, 67, 89, 34, 23, 43, 22]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21, 30, 29, 32, 28, 27, 31, 30, 29, 28, 26]
classes = [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0]

# siapkan kNN
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(list(zip(x, y)), classes)

# simpan titik baru di list agar replot semua
new_points_x = []
new_points_y = []
new_points_class = []

plt.ion()  # mode interaktif
fig, ax = plt.subplots()

def plot_all():
    ax.clear()
    # plot data awal
    ax.scatter(x, y, c=classes, marker='o', label='Data Awal')
    # plot titik baru
    if new_points_x:
        ax.scatter(new_points_x, new_points_y, c=new_points_class,
                   edgecolor='black', s=100, marker='s', label='Input Baru')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('KNN Dynamic Plot (k=1)')
    ax.legend()
    plt.draw()
    plt.pause(0.1)

plot_all()

# Loop input dinamis 
while True:
    user_in = input("Masukkan titik baru (format: x y) atau Enter untuk keluar: ")
    if not user_in.strip():
        break
    try:
        new_x, new_y = map(float, user_in.split())
    except ValueError:
        print("Format salah! Contoh: 8 21")
        continue

    prediction = knn.predict([(new_x, new_y)])[0]
    print(f"Prediksi kelas: {prediction}")

    # simpan dan replot
    new_points_x.append(new_x)
    new_points_y.append(new_y)
    new_points_class.append(prediction)
    plot_all()

plt.ioff()
plt.show()
