"""
Simulating the linear regression two variables using least-square
Creator: Ahmad Sirojuddin
mail: sirojuddin@its.ac.id
"""

import model_train
import matplotlib.pyplot as plt

model_train.set_printoptions(precision=3)
model_train.manual_seed(123)

# ----------- Generating a toy dataset ----------- #
"""
The relation between input and output is stated as y_train(x0, x1) = w0 + w1 * x1 + w2 * x2 + noise
Let's set w0 = 1.2; w1 = 0.6; w2 = 0.8; and noise is gaussian with zero mean and variance = 0.5
"""
w0 = -2.1
w1 = 0.3
w2 = 0.7

x_low = -3
x_high = 7
x1 = model_train.rand(20) * (x_high - x_low) + x_low  # uniform random samples between x_low and x_high
print(f'x1 = {x1}')
x2 = model_train.rand(20) * (x_high - x_low) + x_low  # uniform random samples between x_low and x_high
print(f'x2 = {x2}')
y = w0 + w1 * x1 + w2 * x2 + model_train.randn_like(x1) * model_train.sqrt(model_train.tensor(1.))
print(f'y_train = {y}')

fig = plt.figure()
ax_3D = fig.add_subplot(projection='3d')
ax_3D.scatter(x1, x2, y)

# ----------- Finding the value of w0 and w1 that best fit the scatter ----------- #
# X_train = np.concatenate((np.ones(shape=(x1.size, 1)), np.expand_dims(x1, axis=1), np.expand_dims(x2, axis=1)), axis=1)
X_train = model_train.cat(tensors=(model_train.ones_like(x1).unsqueeze(dim=1),
                             model_train.unsqueeze(x1, dim=1),
                             model_train.unsqueeze(x2, dim=1)),
                    dim=1)
print(f'X_train = {X_train}')
y_train = y.unsqueeze(dim=1)
print(f'y_train = {y_train}')

w_vect = model_train.inverse(X_train.T @ X_train) @ X_train.T @ y_train  # The formula for finding w1 and w0 that best fit the scatter
w0_pred = w_vect[0]
w1_pred = w_vect[1]
w2_pred = w_vect[2]
print(f'w0_pred = {w0_pred} | w1_pred = {w1_pred} | w2_pred = {w2_pred}')
y_pred = w0_pred + w1_pred * x1 + w2_pred * x2

# ----------- Crate a 3D plot ----------- #
X1, X2 = model_train.meshgrid(model_train.arange(start=x_low, end=x_high, step=0.5), model_train.arange(start=x_low, end=x_high, step=0.5),
                        indexing='xy')
Y = w0_pred + w1_pred * X1 + w2_pred * X2
ax_3D.plot_surface(X1, X2, Y, color=(0, 0, 0), alpha=0.5)
ax_3D.set_xlabel('x1')
ax_3D.set_ylabel('x2')
ax_3D.set_zlabel('x3')

plt.show()