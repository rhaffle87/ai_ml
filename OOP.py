class Mahasiswa:
    def __init__(self, nama, NRP, jurusan):
        self.nama = nama
        self.NRP = NRP
        self.jurusan = jurusan
    
    def tampilkan_info(self):
        print("Nama:", self.nama)
        print("NRP:", self.NRP)
        print("Jurusan:", self.jurusan)

# List kosong untuk menyimpan objek mahasiswa
daftar_mahasiswa = []

# Input jumlah mahasiswa
jumlah = int(input("Masukkan jumlah mahasiswa: "))

# Loop untuk memasukkan data mahasiswa
for i in range(jumlah):
    print(f"\nMahasiswa ke-{i+1}:")
    nama = input("Nama: ")
    nrp = input("NRP: ")
    jurusan = input("Jurusan: ")
    
    # Buat objek Mahasiswa dan masukkan ke list
    mhs = Mahasiswa(nama, nrp, jurusan)
    daftar_mahasiswa.append(mhs)

# Tampilkan semua nama mahasiswa
print("\nDaftar Nama Mahasiswa:")
for mhs in daftar_mahasiswa:
    print("-", mhs.nama)
