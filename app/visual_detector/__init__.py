from flask import Blueprint
from .model import MainDetector
from config.dev import SAVE_PATH, BATCH_SIZE, DETECTORS_TO_RUN

# TODO: How do I get SAVE path from Config here?

visual = Blueprint("visual", __name__)
detector = MainDetector(
    save_path=SAVE_PATH,
    batch_size=BATCH_SIZE,
    detectors_to_run=DETECTORS_TO_RUN
)

from app.visual_detector import routes
