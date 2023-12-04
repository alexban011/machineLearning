import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Load data
data = []
with open("points.txt", "r") as f:
    for line in f:
        x, y, _ = line.split()
        data.append([float(x), float(y)])
data = np.array(data)

# Initialize neurons in a grid
x = np.linspace(-300, 300, 10)
y = np.linspace(-300, 300, 10)
neurons = np.array([[np.array([i, j]) for i in x] for j in y])

num_iterations = 100
current_iteration = 0

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)

def update_plot():
    global current_iteration, neurons

    if current_iteration >= num_iterations:
        return

    ax.clear()
    ax.set_xlim(-300, 300)
    ax.set_ylim(-300, 300)
    ax.scatter(data[:, 0], data[:, 1], color='black', s=1)

    for i in range(neurons.shape[0]):
        for j in range(neurons.shape[1]):
            neuron = neurons[i, j]
            ax.plot(neuron[0], neuron[1], 'bo')
            if i > 0:
                ax.plot([neuron[0], neurons[i - 1, j, 0]], [neuron[1], neurons[i - 1, j, 1]], 'b')
            if j > 0:
                ax.plot([neuron[0], neurons[i, j - 1, 0]], [neuron[1], neurons[i, j - 1, 1]], 'b')

    fig.canvas.draw()

    # Train for one iteration
    for point in data:
        diff = neurons - point
        distances = np.linalg.norm(diff, axis=2)
        BMU = np.unravel_index(np.argmin(distances, axis=None), distances.shape)
        delta = 0.2 * (point - neurons[BMU])
        neurons[BMU] += delta
        for i in range(max(0, BMU[0] - 1), min(neurons.shape[0], BMU[0] + 2)):
            for j in range(max(0, BMU[1] - 1), min(neurons.shape[1], BMU[1] + 2)):
                if (i, j) != BMU:
                    neurons[i, j] += delta * 0.5

    current_iteration += 1

# Add button for next iteration
next_ax = plt.axes([0.8, 0.05, 0.1, 0.075])
next_button = Button(next_ax, 'Next')
next_button.on_clicked(lambda event: update_plot())

plt.show()
