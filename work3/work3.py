import numpy as np
import random
import math

# Функция для вычисления расстояний от объекта до базовых станций с добавлением шума
def calculate_distances(object_coords, base_stations_coords, noise_variance):
    distances = []
    for base_coords in base_stations_coords:
        distance = np.linalg.norm(object_coords - base_coords) + np.random.normal(scale=noise_variance)
        distances.append(distance)
    return distances

# Функция для вычисления углов триангуляции
def calculate_triangles(object_coords, base_stations_coords):
    angles = []
    for base_coords in base_stations_coords:
        angle = math.atan2(object_coords[1] - base_coords[1], object_coords[0] - base_coords[0])
        angles.append(angle)
    return angles

# Функция для определения координат объекта с помощью трилатерации
def trilateration(distances, base_stations_coords):
    A = 2 * (base_stations_coords[0][0] - base_stations_coords[2][0])
    B = 2 * (base_stations_coords[0][1] - base_stations_coords[2][1])
    C = 2 * (base_stations_coords[1][0] - base_stations_coords[2][0])
    D = 2 * (base_stations_coords[1][1] - base_stations_coords[2][1])

    E = distances[0]**2 - distances[2]**2 - base_stations_coords[0][0]**2 + base_stations_coords[2][0]**2 - base_stations_coords[0][1]**2 + base_stations_coords[2][1]**2
    F = distances[1]**2 - distances[2]**2 - base_stations_coords[1][0]**2 + base_stations_coords[2][0]**2 - base_stations_coords[1][1]**2 + base_stations_coords[2][1]**2

    x = (E * D - B * F) / (A * D - B * C)
    y = (A * F - E * C) / (A * D - B * C)

    return [x, y]

# Функция для градиентного спуска
def gradient_descent(distances, base_stations_coords, learning_rate, iterations):
    object_coords = np.zeros(2)  # Initialize object coordinates
    for _ in range(iterations):
        gradient = np.zeros(2)
        for i, base_coords in enumerate(base_stations_coords):
            distance_diff = np.linalg.norm(object_coords - base_coords) - distances[i]
            if np.linalg.norm(object_coords - base_coords) != 0:  # Avoid division by zero
                gradient += 2 * distance_diff * (object_coords - base_coords) / np.linalg.norm(object_coords - base_coords)
        object_coords -= learning_rate * gradient
    return object_coords

# Тестирование методов
def main():
    base_stations_coords = np.array([[0, 0], [5, 0], [2.5, 5]])
    object_coords = np.array([3, 3])  # Координаты объекта

    noise_variance = 0.1
    distances = calculate_distances(object_coords, base_stations_coords, noise_variance)
    print("Measured distances with noise:", distances)

    angles = calculate_triangles(object_coords, base_stations_coords)
    print("Triangulation angles:", angles)

    trilateration_coords = trilateration(distances, base_stations_coords)
    print("Object coordinates using trilateration:", trilateration_coords)

    learning_rate = 0.01
    iterations = 1000
    gradient_descent_coords = gradient_descent(distances, base_stations_coords, learning_rate, iterations)
    print("Object coordinates using gradient descent:", gradient_descent_coords)

if __name__ == "__main__":
    main()

