from flask import Flask, render_template, request, jsonify
import core
import core_2D_version2
import core_3D_version2
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('4D printing final version.html')

@app.route('/predict', methods=['POST'])
@app.route('/predict/random', methods=['POST'])
def predict_random():
    data = request.json
    length = int(data['length'])

    input_code = np.random.randint(0, 2, size=(2, length))

    grid_image = core.generate_binary_grid(input_code)
    curve_image = core.generate_plot(input_code)

    return jsonify({
        'grid_image': grid_image,
        'curve_image': curve_image
    })

@app.route('/predict/custom', methods=['POST'])
def predict_custom():
    data = request.json
    input_code = data.get('input_code')

    input_code = np.array(input_code, dtype=int)
    if input_code.shape[0] != 2 or not np.all((input_code == 0) | (input_code == 1)):
        return jsonify({'error': 'Invalid input! Must be 2 rows of binary values.'}), 400

    grid_image = core.generate_binary_grid(input_code)
    curve_image = core.generate_plot(input_code)

    return jsonify({
        'grid_image': grid_image,
        'curve_image': curve_image
    })

@app.route('/predict2D', methods=['POST'])
def predict2D():

    data = request.json
    number = int(data.get('number', 1))
    length = int(data.get('length', 1))

    inputs_total, inputs_angle = core_2D_version2.input_code_angle(number, length)

    random_code_2D = inputs_total[0]
    grid_image_2D = core_2D_version2.generate_binary_grid_2D(random_code_2D)

    image_2D = core_2D_version2.Predict_2D(inputs_total, inputs_angle)

    return jsonify({
        'image_2D': image_2D,
        'grid_image_2D': grid_image_2D
    })

@app.route('/predict2D/uniform', methods=['POST'])
def predict2D_uniform():
    data = request.json
    angles_deg = data.get('angles', [])
    codes = data.get('codes', [])

    if len(angles_deg) != len(codes):
        return jsonify({'error': 'angles and codes mismatch'}), 400

    inputs_total = []
    inputs_angle = []
    for i in range(len(codes)):
        code_i = np.array(codes[i], dtype=int)
        angle_deg = float(angles_deg[i])
        angle_rad = angle_deg * np.pi / 180.0
        inputs_total.append(code_i)
        inputs_angle.append(angle_rad)

    grid_2D = core_2D_version2.generate_binary_grid_2D(inputs_total[0])
    image_2D = core_2D_version2.Predict_2D(inputs_total, inputs_angle)

    return jsonify({'image_2D': image_2D, 'grid_2D': grid_2D})


@app.route('/predict2D/custom', methods=['POST'])
def predict2D_custom():
    data = request.json
    angles_deg = data.get('angles', [])
    codes = data.get('codes', [])

    if len(angles_deg) != len(codes):
        return jsonify({'error': 'angles and codes size mismatch'}), 400

    grids_2D = []
    inputs_total = []
    inputs_angle_radians = []


    for i, code_2d in enumerate(codes):

        code_np = np.array(code_2d, dtype=int)
        angle_in_degrees = float(angles_deg[i])
        angle_in_radians = angle_in_degrees * np.pi / 180.0

        inputs_total.append(code_np)
        inputs_angle_radians.append(angle_in_radians)

        grid_image = core_2D_version2.generate_binary_grid_2D(code_np)
        grids_2D.append(grid_image)

    final_image_2D = core_2D_version2.Predict_2D(inputs_total, inputs_angle_radians)

    return jsonify({
        'image_2D': final_image_2D,
        'grids_2D': grids_2D
    })

@app.route('/predict3D', methods=['POST'])
def predict_3d():
    data = request.json
    number = int(data.get('number', 1))
    length = int(data.get('length', 1))

    inputs_total, inputs_angle = core_3D_version2.input_code_angel(number, length)
    grid_image_3D = core_3D_version2.generate_binary_grid_2D(inputs_total[0])

    result = core_3D_version2.prediction_3D(inputs_total, inputs_angle)

    return jsonify({
        "grid_image_3D": grid_image_3D,
        "image_3D": result["image_3D"],
        "x": result["x"],
        "y": result["y"],
        "z": result["z"]
    })

@app.route('/predict3D/uniform', methods=['POST'])
def predict3D_uniform():
    data = request.json
    angles_deg = data.get('angles', [])
    codes = data.get('codes', [])

    inputs_total = []
    inputs_angle = []
    for i in range(len(codes)):
        code_i = np.array(codes[i], dtype=int)
        angle_rad = float(angles_deg[i]) * np.pi / 180.0
        inputs_total.append(code_i)
        inputs_angle.append(angle_rad)

    grid_image_3D = core_3D_version2.generate_binary_grid_2D(inputs_total[0])
    result = core_3D_version2.prediction_3D(inputs_total, inputs_angle)

    return jsonify({
        "grid_image_3D": grid_image_3D,
        "image_3D": result["image_3D"],
        "x": result["x"],
        "y": result["y"],
        "z": result["z"]
    })

@app.route('/predict3D/custom', methods=['POST'])
def predict_3d_custom():
    data = request.json
    angles_deg = data.get('angles', [])
    codes = data.get('codes', [])

    if len(angles_deg) != len(codes):
        return jsonify({'error': 'angles and codes size mismatch'}), 400

    inputs_total = []
    inputs_angle_radians = []

    for i, code_2d in enumerate(codes):
        code_np = np.array(code_2d, dtype=int)
        angle_rad = float(angles_deg[i]) * np.pi / 180.0
        inputs_total.append(code_np)
        inputs_angle_radians.append(angle_rad)

    grid_image_3D = core_3D_version2.generate_binary_grid_2D(inputs_total[0])
    result = core_3D_version2.prediction_3D(inputs_total, inputs_angle_radians)

    return jsonify({
        "grid_image_3D": grid_image_3D,
        "image_3D": result["image_3D"],
        "x": result["x"],
        "y": result["y"],
        "z": result["z"]
    })


if __name__ == '__main__':
    app.run(debug=True, threaded=False)
