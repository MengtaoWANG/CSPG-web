import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np
import imageio
import ast
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
import scipy.interpolate as spi
import itertools
from scipy.optimize import fsolve, least_squares
from scipy.interpolate import CubicSpline
import os
from scipy.spatial import KDTree
from scipy.signal import correlate
import time
import plotly.graph_objects as go
import plotly.offline as pyo
import base64
import io
import matplotlib.patches as patches
import matplotlib
matplotlib.use('Agg')

from matplotlib.colors import ListedColormap


def load_45points(file_path):
    data = pd.read_csv(file_path, header=None, skiprows=None)
    outputs = []

    for i in range(1, len(data), 2):
        output_up = data.iloc[i - 1, :].to_numpy().astype(float)
        output_down = data.iloc[i, :].to_numpy().astype(float)
        output = np.array([output_up, output_down])
        outputs.append(output.reshape(2, 45))

    return np.array(outputs)

curvature_first_two_rows = pd.read_csv('data/curvature_first_mean.csv', header=None, skiprows=None)
curvature_first_two_rows = np.array(curvature_first_two_rows)
# print(curvature_first_two_rows.shape)
distance_total_mean = pd.read_csv('data/distance_middle_mean.csv', header=None, skiprows=None)
distance_total_mean = np.array(distance_total_mean)
# print(distance_total_mean.shape)
curvature_total_mean = pd.read_csv('data/curvature_middle_mean.csv', header=None, skiprows=None)
curvature_total_mean = np.array(curvature_total_mean)
# print(curvature_total_mean.shape)

curvature_end_mean = pd.read_csv('data/curvature_last_mean.csv', header=None, skiprows=None)
curvature_end_mean = np.array(curvature_end_mean)

distance_end_mean = pd.read_csv('data/distance_last_mean.csv', header=None, skiprows=None)
distance_end_mean = np.array(distance_end_mean)


coordinates_45points_file = 'data/coordinates_45points_end.csv'
coordinates_45points = load_45points(coordinates_45points_file)


def best_fit_transform(A, B):
    assert A.shape == B.shape

    # Calculate the centroids of the point sets
    centroid_A = np.mean(A, axis=1).reshape(2, 1)
    centroid_B = np.mean(B, axis=1).reshape(2, 1)

    # Center the points
    AA = A - centroid_A
    BB = B - centroid_B

    # Compute the covariance matrix
    H = BB @ AA.T

    # Compute the Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H)

    # Compute the rotation matrix
    R = Vt.T @ U.T

    # Handle the reflection case (det(R) < 0)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute the translation
    t = centroid_A - R @ centroid_B

    return R, t



def compute_curvature_spline(x, y):

    tck, u = spi.splprep([x, y], s=0, k=2)
    # print('u', u)
    dx, dy = spi.splev(u, tck, der=1)
    ddx, ddy = spi.splev(u, tck, der=2)

    curvature = (dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** (3 / 2)

    return curvature, u

def calculate_third_point(p1, p2, distance, curvature):
    x1, y1 = p1
    x2, y2 = p2
    L12 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def area(x, y):
        return 0.5 * np.abs(x1 * (y2 - y) + x2 * (y - y1) + x * (y1 - y2))

    def equations(p):
        x, y = p
        L31 = np.sqrt((x - x1)**2 + (y - y1)**2)
        A = area(x, y)
        eq1 = A - (np.abs(curvature) * L12 * distance * L31 / 4)
        eq2 = (x - x2) ** 2 + (y - y2) ** 2 - distance ** 2

        if not all(np.isfinite([x, y, eq1, eq2])):
            raise ValueError(
                f"Non-finite value encountered in equations: x={x}, y={y}, eq1={eq1}, eq2={eq2}")

        return [eq1, eq2]

    initial_guesses = [
        [x2 + distance, y2],
        [x2 - distance, y2],
        [x2, y2 + distance],
        [x2, y2 - distance],
        [x2 + distance, y2 + distance],
        [x2 + distance, y2 - distance],
        [x2 - distance, y2 + distance],
        [x2 - distance, y2 - distance],
    ]

    solutions = []
    for guess in initial_guesses:
        try:
            solution = fsolve(equations, guess)
            solutions.append(solution)
        except Exception as e:
            print(f"Failed to converge with initial guess {guess}: {e}")
    # print('solutions', solutions)
    best_solution = None
    best_residual = float('inf')
    solutions1 = np.array(solutions)

    three_curvature = []
    for solution in solutions:
        x3, y3 = solution
        xs = np.array([x1, x2, x3])
        ys = np.array([y1, y2, y3])

        curvatures, _ = compute_curvature_spline(xs, ys)
        interpolated_curvature = curvatures[1]
        # print('interpolated_curvature', interpolated_curvature)
        residual = np.abs(interpolated_curvature - curvature)
        # print('residual', residual)
        three_curvature.append(interpolated_curvature)
        if residual < best_residual:
            best_residual = residual
            # print('best_residual', best_residual)
            best_solution = solution
            # print('best_solution', best_solution)

    if best_solution is not None:
        return np.array(best_solution)
    else:
        raise ValueError("No valid solution found")

def z_normalize(ts):
    return (ts - np.mean(ts)) / np.std(ts)


patterns = [
    np.array([0., 0.]),
    np.array([0., 1.]),
    np.array([1., 0.]),
    np.array([1., 1.])
]

all_combinations = list(itertools.product([0, 1], repeat=6))

all_codes = np.array([np.array(combination).reshape(2, 3) for combination in all_combinations])
# print(all_codes)

def three_dimensional_prediction(x, angle):
    len_x = x.shape[-1]
    x_point1 = np.concatenate([np.full((1, len_x), x[0, 0]), np.full((1, len_x), x[1, 0]), np.full((1, len_x), x[2, 0])], axis=0)
    new_x = x - x_point1
    rotation_matrix_ni = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])

    final_x = np.dot(rotation_matrix_ni, new_x)

    return final_x


def input_code_angel(number, length):
    input_total_code = []
    input_total_angle = []
    input_encoding = np.random.randint(0, 2, size=(2, length))
    for i in range(number):

        input_angle = i * (2 * np.pi) / number
        input_total_code.append(input_encoding)
        input_total_angle.append(input_angle)

    return input_total_code, input_total_angle

def generate_binary_grid_2D(input_code):

    cmap = ListedColormap(['#77CCC3', '#E96333'])

    cell_size = 0.5
    rows, cols = input_code.shape

    fig, ax = plt.subplots(figsize=(cols * cell_size, rows * cell_size))
    cax = ax.matshow(input_code, cmap=cmap)

    for i in range(rows):
        for j in range(cols):
            rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     linewidth=1, edgecolor='white', facecolor='none')
            ax.add_patch(rect)
            ax.text(j, i, input_code[i, j],
                    ha='center', va='center', color='white', fontsize=10)

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)

    return base64.b64encode(buf.getvalue()).decode('utf-8')


def prediction_3D(inputs_total, inputs_angle):

    output_total = []

    for i in range(len(inputs_total)):
        input_code = inputs_total[i]
        input_len = input_code.shape[1]
        print(i, 'input_code', input_code)

        if input_len == 1:
            new_columns = np.tile(input_code[:, -1].reshape(2, 1), (1, 2))
            input_expand = np.hstack((input_code, new_columns))
            for index, first_three_encoding in enumerate(all_codes):
                if np.all(input_expand[:, 0:3] == first_three_encoding):
                    # print(index, input_code[:, 0:3])
                    coordinate_original_points = coordinates_45points[index]
                    coordinate_original_points = coordinate_original_points[:, 0:20]
        if input_len == 2:
            new_columns = np.tile(input_code[:, -1].reshape(2, 1), (1, 1))
            input_expand = np.hstack((input_code, new_columns))
            for index, first_three_encoding in enumerate(all_codes):
                if np.all(input_expand[:, 0:3] == first_three_encoding):
                    # print(index, input_code[:, 0:3])
                    coordinate_original_points = coordinates_45points[index]
                    coordinate_original_points = coordinate_original_points[:, 0:40]

        if input_len == 3:
            new_columns = np.tile(input_code[:, -1].reshape(2, 1), (1, 1))
            input_expand = np.hstack((input_code, new_columns))
            for index, first_three_encoding in enumerate(all_codes):
                if np.all(input_expand[:, 0:3] == first_three_encoding):
                    # print(index, input_code[:, 0:3])
                    coordinate_original_points = coordinates_45points[index]

            for ii in range(1, input_len-3):
                # print('=====================================================')
                for index, subsequent_code in enumerate(all_codes):
                    if np.all(input_expand[:, ii:ii+3] == subsequent_code):
                        # print('index',index, input_code[:, ii:ii+3])
                        distance_index_data = distance_total_mean[index]
                        curvature_index_data = curvature_total_mean[index]
                        for index in range(0, 20):

                            distance_mean_index = distance_index_data[index]
                            curvature_mean_index = curvature_index_data[index]
                            # print('distance_mean_index, curvature_mean_index', distance_mean_index, curvature_mean_index)
                            new_point = calculate_third_point(coordinate_original_points[:, -2], coordinate_original_points[:, -1], distance_mean_index, curvature_mean_index)
                            coordinate_original_points = np.column_stack((coordinate_original_points, new_point))

            for index, end_encoding in enumerate(all_codes):
                if np.all(input_expand[:, -3:] == end_encoding):
                    # print('-------------------------', index, input_code[:, -3:0])
                    distance_end_mean_data = distance_end_mean[index]
                    # print(distance_end_mean_data.shape)
                    curvature_end_mean_data = curvature_end_mean[index]
                    for index in range(0, 34):
                        # print('index', index)

                        distance_index = distance_end_mean_data[index]
                        curvature_index = curvature_end_mean_data[index]
                        # print('distance_index, curvature_index', distance_index, curvature_index)
                        new_point = calculate_third_point(coordinate_original_points[:, -2], coordinate_original_points[:, -1], distance_index, curvature_index)
                        coordinate_original_points = np.column_stack((coordinate_original_points, new_point))
                        coordinate_original_points = coordinate_original_points[:, 0:60]

        if input_len > 3:

            for index, first_three_encoding in enumerate(all_codes):
                if np.all(input_code[:, 0:3] == first_three_encoding):
                    # print(index, input_code[:, 0:3])
                    coordinate_original_points = coordinates_45points[index]

            for ii in range(1, input_len-3):
                # print('=====================================================')
                for index, subsequent_code in enumerate(all_codes):
                    if np.all(input_code[:, ii:ii+3] == subsequent_code):
                        # print('index',index, input_code[:, ii:ii+3])
                        distance_index_data = distance_total_mean[index]
                        curvature_index_data = curvature_total_mean[index]
                        for index in range(0, 20):

                            distance_mean_index = distance_index_data[index]
                            curvature_mean_index = curvature_index_data[index]
                            # print('distance_mean_index, curvature_mean_index', distance_mean_index, curvature_mean_index)
                            new_point = calculate_third_point(coordinate_original_points[:, -2], coordinate_original_points[:, -1], distance_mean_index, curvature_mean_index)
                            coordinate_original_points = np.column_stack((coordinate_original_points, new_point))

            for index, end_encoding in enumerate(all_codes):
                if np.all(input_code[:, -3:] == end_encoding):
                    # print('-------------------------', index, input_code[:, -3:0])
                    distance_end_mean_data = distance_end_mean[index]
                    # print(distance_end_mean_data.shape)
                    curvature_end_mean_data = curvature_end_mean[index]
                    for index in range(0, 34):
                        # print('index', index)

                        distance_index = distance_end_mean_data[index]
                        curvature_index = curvature_end_mean_data[index]
                        # print('distance_index, curvature_index', distance_index, curvature_index)
                        new_point = calculate_third_point(coordinate_original_points[:, -2], coordinate_original_points[:, -1], distance_index, curvature_index)
                        coordinate_original_points = np.column_stack((coordinate_original_points, new_point))
                        # print(coordinate_original_points.shape)
                        # print(coordinate_original_points)
        coordinate_original_points = coordinate_original_points / 4.
        y_length = coordinate_original_points.shape[1]
        y_axis_data = np.zeros((1, y_length))
        coordinates_3d = np.vstack((coordinate_original_points[0], y_axis_data[0], coordinate_original_points[1]))

        output_total.append(coordinates_3d)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    all_x = []
    all_y = []
    all_z = []

    for i in range(len(output_total)):
        output_total_i = three_dimensional_prediction(output_total[i], inputs_angle[i])

        x_data = output_total_i[0]
        y_data = output_total_i[1]
        z_data = output_total_i[2]

        all_x.extend(x_data)
        all_y.extend(y_data)
        all_z.extend(z_data)

        ax.plot(x_data, y_data, z_data, marker='o', linestyle='-', markersize=3)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.view_init(elev=45, azim=45)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    plt.close()

    return {
        "image_3D": base64.b64encode(buf.getvalue()).decode('utf-8'),
        "x": all_x,
        "y": all_y,
        "z": all_z
    }
























