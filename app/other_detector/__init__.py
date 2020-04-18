from flask import Blueprint

other = Blueprint(name="other", import_name=__name__)

from app.other_detector import routes
