import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def k_means(points, k, tol, max_iter):
    centroids = []

    # Randomly initialize centroids
    random_points = random.sample(list(points), k)
    for point in random_points:
        centroids.append(point)

    for i in range(max_iter):
        classes = {}
        for j in range(k):
            classes[j] = []

        # Assign points to nearest centroid
        for point in points:
            distances = [np.linalg.norm(point - centroid) for centroid in centroids]
            nearest_centroid = distances.index(min(distances))
            classes[nearest_centroid].append(point)

        # Save old centroids for convergence check
        old_centroids = centroids.copy()

        # Calculate new centroids
        for j in range(k):
            centroids[j] = np.average(classes[j], axis=0)

        # Plot points and Centroid
        colors = 10 * ['r', 'g', 'b', 'c', 'k', 'y', 'm']
        for j in range(k):
            color = colors[j]
            features = np.array(classes[j])
            plt.scatter(features[:, 0], features[:, 1], color=color, s=1)
            plt.scatter(centroids[j][0], centroids[j][1], s=300, c="black")

        plt.show()
        if i < max_iter - 1:
            plt.pause(1)  # Pause for 2 seconds
            plt.clf()  # Clear the figure for the next plot

        plt.show()

        # Check if we have converged
        isOptimal = True

        for j in range(k):
            original_centroid = old_centroids[j]
            current_centroid = centroids[j]

            if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > tol:
                isOptimal = False

                # If centroids don't change significantly, we can stop
        if isOptimal:
            break

    return centroids, classes


def main():
    df = pd.read_csv('points.txt', sep=' ', header=None)
    df[[0, 1]] = df[[0, 1]].apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    points = df[[0, 1]].to_numpy()

    k = random.randint(2, 10)  # Define number of clusters
    tol = 0.001  # Define tolerance
    max_iter = 10  # Define maximum iterations

    print(f"Centroids: ${k}")

    centroids, classes = k_means(points, k, tol, max_iter)  # Run k-means

    cost = 0
    for centroid_index in classes:
        for point in classes[centroid_index]:
            cost += np.linalg.norm(centroids[centroid_index] - point)

    print('Cost of the model:', cost)


if __name__ == "__main__":
    main()