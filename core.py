from flask import Flask, request, render_template, jsonify
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
import io
import base64
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.colors import ListedColormap
import io
import matplotlib.patches as patches
import matplotlib
matplotlib.use('Agg')


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


def compute_curvature_spline(x, y):
    # 使用样条插值
    tck, u = spi.splprep([x, y], s=0, k=2)
    # print('u', u)
    dx, dy = spi.splev(u, tck, der=1)
    ddx, ddy = spi.splev(u, tck, der=2)

    # 计算曲率
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
        [x2 - distance, y2 - distance]
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
    # print(solutions1.shape)

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


patterns = [
    np.array([0., 0.]),
    np.array([0., 1.]),
    np.array([1., 0.]),
    np.array([1., 1.])
]

all_combinations = list(itertools.product([0, 1], repeat=6))

all_codes = np.array([np.array(combination).reshape(2, 3) for combination in all_combinations])


def inputs_code(length):
    return np.random.randint(0, 2, size=(2, length))



def generate_binary_grid(input_code):
    # cmap = ListedColormap(['#E96333', '#77CCC3'])
    cmap = ListedColormap(['#77CCC3', '#E96333'])

    cell_size = 0.5
    rows, cols = input_code.shape
    fig_block, ax_block = plt.subplots(figsize=(cols * cell_size, rows * cell_size))
    cax = ax_block.matshow(input_code, cmap=cmap)

    for i in range(input_code.shape[0]):
        for j in range(input_code.shape[1]):
            rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=1, edgecolor='white', facecolor='none')
            ax_block.add_patch(rect)
            ax_block.text(j, i, input_code[i, j], ha='center', va='center', color='white', fontsize=10)

    ax_block.set_xlim(-0.5, input_code.shape[1] - 0.5)
    ax_block.set_ylim(input_code.shape[0] - 0.5, -0.5)
    ax_block.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)

    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_plot(input_code):
    input_len = input_code.shape[1]
    if input_len == 1:
        new_columns = np.tile(input_code[:, -1].reshape(2, 1), (1, 2))
        input_expand = np.hstack((input_code, new_columns))
        # print(input_expand)
        for index, first_three_encoding in enumerate(all_codes):
            if np.all(input_expand[:, 0:3] == first_three_encoding):
                # print(index, input_code[:, 0:3])
                coordinate_original_points = coordinates_45points[index]
                coordinate_original_points = coordinate_original_points[:, 0:20]
    if input_len == 2:
        new_columns = np.tile(input_code[:, -1].reshape(2, 1), (1, 1))
        input_expand = np.hstack((input_code, new_columns))
        # print(input_expand)
        for index, first_three_encoding in enumerate(all_codes):
            if np.all(input_expand[:, 0:3] == first_three_encoding):
                # print(index, input_code[:, 0:3])
                coordinate_original_points = coordinates_45points[index]
                coordinate_original_points = coordinate_original_points[:, 0:40]

    if input_len == 3:
        new_columns = np.tile(input_code[:, -1].reshape(2, 1), (1, 1))
        input_expand = np.hstack((input_code, new_columns))
        # print(input_expand)
        for index, first_three_encoding in enumerate(all_codes):
            if np.all(input_expand[:, 0:3] == first_three_encoding):
                # print(index, input_code[:, 0:3])
                coordinate_original_points = coordinates_45points[index]

        for ii in range(1, input_len - 3):
            # print('=====================================================')
            for index, subsequent_code in enumerate(all_codes):
                if np.all(input_expand[:, ii:ii + 3] == subsequent_code):
                    # print('index',index, input_code[:, ii:ii+3])
                    distance_index_data = distance_total_mean[index]
                    curvature_index_data = curvature_total_mean[index]
                    for index in range(0, 20):
                        distance_mean_index = distance_index_data[index]
                        curvature_mean_index = curvature_index_data[index]
                        # print('distance_mean_index, curvature_mean_index', distance_mean_index, curvature_mean_index)
                        new_point = calculate_third_point(coordinate_original_points[:, -2],
                                                          coordinate_original_points[:, -1], distance_mean_index,
                                                          curvature_mean_index)
                        coordinate_original_points = np.column_stack((coordinate_original_points, new_point))

        for index, end_encoding in enumerate(all_codes):
            if np.all(input_expand[:, -3:] == end_encoding):
                # print('-------------------------', index, input_code[:, -3:0])
                distance_end_mean_data = distance_end_mean[index]
                # print(distance_end_mean_data.shape)
                curvature_end_mean_data = curvature_end_mean[index]
                for index in range(0, 34):
                    distance_index = distance_end_mean_data[index]
                    curvature_index = curvature_end_mean_data[index]
                    # print('distance_index, curvature_index', distance_index, curvature_index)
                    new_point = calculate_third_point(coordinate_original_points[:, -2],
                                                      coordinate_original_points[:, -1], distance_index,
                                                      curvature_index)
                    coordinate_original_points = np.column_stack((coordinate_original_points, new_point))
                    coordinate_original_points = coordinate_original_points[:, 0:60]

    if input_len > 3:

        for index, first_three_encoding in enumerate(all_codes):
            if np.all(input_code[:, 0:3] == first_three_encoding):
                # print(index, input_code[:, 0:3])
                coordinate_original_points = coordinates_45points[index]

        for ii in range(1, input_len - 3):
            # print('=====================================================')
            for index, subsequent_code in enumerate(all_codes):
                if np.all(input_code[:, ii:ii + 3] == subsequent_code):
                    # print('index',index, input_code[:, ii:ii+3])
                    distance_index_data = distance_total_mean[index]
                    curvature_index_data = curvature_total_mean[index]
                    for index in range(0, 20):
                        distance_mean_index = distance_index_data[index]
                        curvature_mean_index = curvature_index_data[index]
                        # print('distance_mean_index, curvature_mean_index', distance_mean_index, curvature_mean_index)
                        new_point = calculate_third_point(coordinate_original_points[:, -2],
                                                          coordinate_original_points[:, -1], distance_mean_index,
                                                          curvature_mean_index)
                        coordinate_original_points = np.column_stack((coordinate_original_points, new_point))

        for index, end_encoding in enumerate(all_codes):
            if np.all(input_code[:, -3:] == end_encoding):
                # print('-------------------------', index, input_code[:, -3:0])
                distance_end_mean_data = distance_end_mean[index]
                # print(distance_end_mean_data.shape)
                curvature_end_mean_data = curvature_end_mean[index]
                for index in range(0, 34):
                    distance_index = distance_end_mean_data[index]
                    curvature_index = curvature_end_mean_data[index]
                    # print('distance_index, curvature_index', distance_index, curvature_index)
                    new_point = calculate_third_point(coordinate_original_points[:, -2],
                                                      coordinate_original_points[:, -1], distance_index,
                                                      curvature_index)
                    coordinate_original_points = np.column_stack((coordinate_original_points, new_point))

    coordinate_original_points = coordinate_original_points / 4.

    plt.plot(coordinate_original_points[0], coordinate_original_points[1], marker='o', linestyle='-', markersize=3,
             label='Predicted deformation')
    # plt.axvline(x=test_output_data[0, 400], color='r', linestyle='--', label='Vertical Line at Point 400')
    plt.xlabel('Coordinate x (m)', fontsize=12)
    plt.ylabel('Coordinate y (m)', fontsize=12)
    plt.grid(False)
    plt.axis('equal')
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    plt.close()

    return base64.b64encode(buf.getvalue()).decode('utf-8')
