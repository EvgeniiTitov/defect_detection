from model import MainDetector
from flask import Flask, jsonify, request

app = Flask(__name__)

# Location on the server where processed images will be saved
SAVE_PATH = r"D:\Desktop\system_output\OUTPUT"

detector = MainDetector(save_path=SAVE_PATH,
                        search_defects=True)


@app.route('/predict', methods=["POST"])
def predict():

    response = {"success": False}

    if request.method == "POST":
        data = request.get_json()

        path_to_data = data["path_to_data"]
        pole_number = data["pole_number"]

        # TODO: Do we need try except here?
        defects = detector.predict(path_to_data=path_to_data,
                                   pole_number=pole_number)

        response["results"] = defects
        response["success"] = True

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
