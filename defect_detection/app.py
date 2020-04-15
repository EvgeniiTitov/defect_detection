from model import MainDetector
from flask import Flask, jsonify, request, abort

app = Flask(__name__)

# Location on the server where processed images will be saved
SAVE_PATH = r"D:\Desktop\system_output\OUTPUT"

detector = MainDetector(save_path=SAVE_PATH,
                        search_defects=True)


@app.route('/predict', methods=["POST"])
def predict():
    """

    :return:
    """
    # TODO: Check if threads and NN are ready

    response = {"success": False}
    data = request.get_json()
    try:
        path_to_data = data["path_to_data"]
        pole_number = data["pole_number"]
        request_id = data["request_id"]
    except Exception as e:
        print(f"Exception raised while reading items sent in request. Error: {e}")
        response["msg"] = "Wrong input. Ensure path to data, pole number and request ID provided"
        return jsonify(response)

    try:
        ids = detector.predict(
            path_to_data=path_to_data,
            pole_number=pole_number,
            request_id=request_id
        )
    except:
        print("\nAttempt to process the files provided failed")
        return jsonify(response)

    response["ids"] = ids
    response["success"] = True

    return jsonify(response)


@app.route('/status', methods=["GET"])
def status():
    data = request.get_json()
    id = data["id"]

    if id in detector.progress:
        progress = {key: detector.progress[id].get(key) for key in ["status",
                                                                    "total_frames",
                                                                    "processed"]}
        return jsonify(progress)

    return abort(404)


@app.route('/shutdown')
def shut_down_server():
    #TODO: Anyone can kill it
    #Kill threads and NN, not actual server

    detector.stop()

    return jsonify({"msg": "Server successfully shut down"})


if __name__ == "__main__":
    app.run(debug=False)
