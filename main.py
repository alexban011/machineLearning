import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def k_means(points, k, tol, max_iter):
    centroids = []
    random_points = []
    currIter = 0

    for i in range(k):
        random_points.append([random.uniform(-300, 300),random.uniform(-300, 300)])

    for point in random_points:
        centroids.append(point)

    for i in range(max_iter):
        currIter = i
        classes = {}
        for j in range(k):
            classes[j] = []

        for point in points:
            distances = [np.linalg.norm(point - centroid) for centroid in centroids]
            nearest_centroid = distances.index(min(distances))
            classes[nearest_centroid].append(point)

        old_centroids = centroids.copy()

        for j in range(k):
            centroids[j] = np.average(classes[j], axis=0)

        colors = 10 * ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        for j in range(k):
            color = colors[j]
            features = np.array(classes[j])
            plt.scatter(features[:, 0], features[:, 1], color=color, s=1)
            plt.scatter(centroids[j][0], centroids[j][1], s=100, c="black")

        plt.show()

        isOptimal = True
        for j in range(k):
            original_centroid = old_centroids[j]
            current_centroid = centroids[j]
            if np.sum((current_centroid - original_centroid)) > tol:
                isOptimal = False

        if isOptimal:
            return currIter

        if i < max_iter - 1:
            plt.pause(1)
            plt.clf()
        plt.show()


def main():
    df = pd.read_csv('points.txt', sep=' ', header=None)
    df[[0, 1]] = df[[0, 1]].apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    points = df[[0, 1]].to_numpy()

    k = random.randint(2, 10)
    tol = 2
    max_iter = 100

    print(f"Centroids: {k}")

    print(f"Iterations: {k_means(points, k, tol, max_iter)}")

if __name__ == "__main__":
    main()