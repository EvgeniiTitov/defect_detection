from app.magnetic_detector import magnetic
from flask import request, jsonify


@magnetic.route("/magnetic/process_batch", methods=["POST"])
def predict():
    response = {"success": False}
    data = request.get_json()

    response["msg"] = data
    response["success"] = True

    return jsonify(response)
