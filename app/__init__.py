from flask import Flask
import os


def create_app(config_type):
    """
    App factory function - build and returns the app
    :param config_type:
    :return:
    """
    # Initialize the flask app
    app = Flask(__name__)
    path_to_config = os.path.join(os.getcwd(), "config", config_type + ".py")
    app.config.from_pyfile(path_to_config)

    # Import and register blueprints
    from app.visual_detector import visual
    app.register_blueprint(visual)

    from app.magnetic_detector import magnetic
    app.register_blueprint(magnetic)

    from app.other_detector import other
    app.register_blueprint(other)

    return app
