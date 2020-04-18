from flask import jsonify, request, abort
from app.visual_detector import visual, detector
#TODO: Routes need detector instance, where to instantiate it?

@visual.route('/predict', methods=["POST"])
def predict():
    # TODO: Check if threads and NN are ready (running)

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


@visual.route('/status', methods=["GET"])
def status():
    data = request.get_json()
    id = data["id"]

    if id in detector.progress:
        progress = {key: detector.progress[id].get(key) for key in ["status",
                                                                    "total_frames",
                                                                    "processed"]}
        return jsonify(progress)

    return abort(404)


@visual.route('/shutdown')
def shut_down_server():
    """
    TODO: Some sort of key or something to be able to stop the threads
    :return:
    """
    detector.stop()

    return jsonify({"msg": "Workers successfully killed"})
