import numpy as np
import matplotlib.pyplot as plt

# Parametrii zonelor
zones = [
    {"m_x": -150, "m_y": 150, "sigma_x": 30, "sigma_y": 10, "color": "red"},
    {"m_x": 0, "m_y": 0, "sigma_x": 30, "sigma_y": 50, "color": "blue"},
    {"m_x": 200, "m_y": -150, "sigma_x": 50, "sigma_y": 20, "color": "yellow"}
]

PRAG = 0.5
OUTLIER_RATIO = 0.001

np.random.seed(None)

def G(x, m, sigma):
    return np.exp(-(m-x)**2 / (2 * sigma**2))

def generate_points_fast(n_points=10000):
    points = []

    while len(points) < n_points:
        zone = np.random.choice(zones)

        x_vals = np.random.uniform(-300, 300, n_points)
        y_vals = np.random.uniform(-300, 300, n_points)

        x_accepted = x_vals[G(x_vals, zone["m_x"], zone["sigma_x"]) > PRAG]
        y_accepted = y_vals[G(y_vals, zone["m_y"], zone["sigma_y"]) > PRAG]

        combined = list(zip(x_accepted, y_accepted))
        points.extend([(x[0], x[1], zone["color"]) for x in combined])

    return points[:n_points]

# Adăugăm outlierii care aparțin zonelor
def add_zone_outliers(points, n_points=10000):
    n_outliers = int(n_points * OUTLIER_RATIO)
    outliers = []

    for zone in zones:
        outliers.extend([
            (np.random.uniform(-300, 300), np.random.uniform(-300, 300), zone["color"])
            for _ in range(n_outliers)
        ])

    return points + outliers

# Salvarea punctelor în fișier
def save_to_file(points, filename="points.txt"):
    with open(filename, "w") as file:
        for point in points:
            file.write(f"{point[0]} {point[1]} {point[2]}\n")

# Afișarea punctelor din fișier
def plot_from_file(filename="points.txt"):
    data = np.loadtxt(filename, dtype={'names': ('x', 'y', 'color'), 'formats': ('f8', 'f8', 'U10')})
    x = data['x']
    y = data['y']
    colors = data['color']

    for zone in zones:
        plt.scatter(x[colors == zone["color"]], y[colors == zone["color"]], color=zone["color"], s=5, marker=',')

# Generăm punctele, le salvăm și le afișăm
points = generate_points_fast()
points_with_outliers = add_zone_outliers(points)
save_to_file(points_with_outliers)
plot_from_file()

plt.axvline(0, c='black', ls='-')
plt.axhline(0, c='black', ls='-')
plt.xlim([-300, 300])
plt.ylim([-300, 300])
plt.show()