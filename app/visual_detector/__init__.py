from flask import Blueprint
from .model import MainDetector
from config.dev import SAVE_PATH

# TODO: How do I get SAVE path from Config here?

visual = Blueprint("visual", __name__)
detector = MainDetector(save_path=SAVE_PATH)

from app.visual_detector import routes
