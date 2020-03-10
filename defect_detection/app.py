from model import MainDetector
from flask import Flask, jsonify, request


app = Flask(__name__)


@app.route('/predict', methods=["POST"])
def predict():

    data = {"success": False}

    if request.method == "POST":
        data = request.get_json()

        path_to_data = data["path_to_data"]
        pole_number = data["pole_number"]

        defects = detector.predict(path_to_data=path_to_data,
                                   pole_number=pole_number)

        data["predictions"] = defects
        data["success"] = True

    return jsonify(data)


if __name__ == "__main__":

    # Some place on the server where processed files will be stored
    SAVE_PATH = r"D:\Desktop\system_output\API_RESULTS"

    detector = MainDetector(save_path=SAVE_PATH)

    app.run(debug=True)
