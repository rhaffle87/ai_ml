from sklearn.model_selection import train_test_split
import numpy as np

# data contoh
data = np.arange(10)  # [0, 1, 2, ..., 9]

# 1️⃣ random_state = 42
train1, test1 = train_test_split(data, test_size=0.3, random_state=42)
train2, test2 = train_test_split(data, test_size=0.3, random_state=42)

print("Random state 42:")
print("Test1:", test1)
print("Test2:", test2)  # sama persis dengan test1

# 2️⃣ random_state = 123
train3, test3 = train_test_split(data, test_size=0.3, random_state=123)
train4, test4 = train_test_split(data, test_size=0.3, random_state=123)

print("\nRandom state 123:")
print("Test3:", test3)
print("Test4:", test4)  # sama persis dengan test3

# 3️⃣ tanpa random_state (acak setiap run)
train5, test5 = train_test_split(data, test_size=0.3)
train6, test6 = train_test_split(data, test_size=0.3)

print("\nTanpa random_state:")
print("Test5:", test5)
print("Test6:", test6)  # bisa berbeda dengan test5
