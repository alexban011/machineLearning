import numpy as np
import matplotlib.pyplot as plt

grid_size = 10
x = np.linspace(-300, 300, 10)
y = np.linspace(-300, 300, 10)
neurons = np.array([[np.array([i, j]) for i in x] for j in y])

learning_rate = 1

with open("points.txt", 'r') as f:
 points = [list(map(float, line.strip().split()[:2])) for line in f]

iterations = len(points)

fig, ax = plt.subplots()
ax.set_xlim(-300, 300)
ax.set_ylim(-300, 300)

for i in range(iterations):
 x = np.array(points[i])

 distances = np.linalg.norm(neurons - x, axis=2)

 bmu_index = np.unravel_index(np.argmin(distances), distances.shape)

 for i in range(max(0, bmu_index[0] - 1), min(grid_size, bmu_index[0] + 2)):
    for j in range(max(0, bmu_index[1] - 1), min(grid_size, bmu_index[1] + 2)):
        neurons[i, j] += learning_rate * (x - neurons[i, j])

 learning_rate *= 0.99

 ax.clear()
 ax.scatter(neurons[:, :, 0], neurons[:, :, 1], s=10)
 ax.scatter(x[0], x[1], c='blue')
 ax.set_xlim(-300, 300)
 ax.set_ylim(-300, 300)
 for i in range(grid_size):
    for j in range(grid_size):
        if i < grid_size - 1:
            ax.plot([neurons[i, j, 0], neurons[i + 1, j, 0]], [neurons[i, j, 1], neurons[i + 1, j, 1]], 'k-')
        if j < grid_size - 1:
            ax.plot([neurons[i, j, 0], neurons[i, j + 1, 0]], [neurons[i, j, 1], neurons[i, j + 1, 1]], 'k-')
 plt.draw()
 plt.pause(0.01)

plt.show()
