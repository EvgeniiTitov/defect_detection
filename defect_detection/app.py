from model import MainDetector
from flask import Flask, jsonify, request, abort
import numpy as np
import cv2

app = Flask(__name__)

# TODO: Might need a separate endpoint to cause inclination detection
#       and one method for defect detection

# Location on the server where processed images will be saved
SAVE_PATH = r"D:\Desktop\system_output\OUTPUT"

detector = MainDetector(save_path=SAVE_PATH,
                        search_defects=True)


@app.route('/predict', methods=["POST"])
def predict():
    response = {"success": False}

    data = request.get_json()

    path_to_data = data["path_to_data"]
    pole_number = data["pole_number"]

    try:
        defects = detector.predict(path_to_data=path_to_data,
                                   pole_number=pole_number)
    except:
        print("\nAttempt to process the files provided failed")
        return jsonify(response)

    response["results"] = defects
    response["success"] = True

    return jsonify(response)


@app.route('/status/{id}', methods=["GET"])
def status(id):
    if id in detector.progress:
        return jsonify(detector.progress[id])

    return abort(404)


@app.route('/process_image', methods=["POST"])
def process_image():
    file = request.files["image"].read()
    np_image = np.fromstring(file, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    # TODO: run inference - create new endpoint to process a single image
    # TODO: return results


if __name__ == "__main__":
    # TODO: Can we change timeout?
    app.run(debug=False)
