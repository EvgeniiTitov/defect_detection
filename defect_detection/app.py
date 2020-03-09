from driver_old import MainDetector
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/predict', methods=["POST"])
def predict():

    data = request.get_json()

    path_to_data = data["path"]
    # SEND POLE'S NUMBER
    pole_number = data["pole_number"]

    defects = detector.parse_input_data(path_to_data=path_to_data,
                                        pole_number=pole_number)

    output = {"defects": defects}

    return output

if __name__ == "__main__":
    # Some place on the server
    SAVE_PATH = r"D:\Desktop\system_output\API_RESULTS"

    detector = MainDetector(save_path=SAVE_PATH)

    app.run(debug=True)
