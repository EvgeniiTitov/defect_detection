from flask import Blueprint


magnetic = Blueprint(name="magnetic", import_name=__name__)

from app.magnetic_detector import routes
