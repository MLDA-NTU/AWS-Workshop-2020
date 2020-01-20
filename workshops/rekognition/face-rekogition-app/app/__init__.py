import os
import logging
from logging.handlers import TimedRotatingFileHandler

from flask import Flask


# Define Flask WSGI Application
flask_app = Flask(__name__,
                  static_url_path='',
                  static_folder='web/static',
                  template_folder='web/templates')

# Application configurations
flask_app.config.from_object('config.flask_config')

# with and without trailing is same
flask_app.url_map.strict_slashes = False


def get_registered_blueprint():
    # Import blueprints here ...
    from app.views import mod_video_stream

    return [
        # get all defined blueprints
        mod_video_stream,
    ]


def setup_logging():
    if not os.path.isdir("logs"):
        os.makedirs('logs')

    formatter = logging.Formatter("[%(asctime)s] %(message)s")

    handler_info = TimedRotatingFileHandler(
        './logs/info.log', when='midnight', interval=1, backupCount=5)
    handler_info.setLevel(logging.INFO)
    handler_info.setFormatter(formatter)

    handler_error = TimedRotatingFileHandler(
        './logs/error.log', when='midnight', interval=1, backupCount=5)
    handler_error.setLevel(logging.ERROR)
    handler_error.setFormatter(formatter)

    # access log
    logger = logging.getLogger('werkzeug')
    handler_access = logging.FileHandler('./logs/access.log')

    flask_app.logger.addHandler(handler_info)
    flask_app.logger.addHandler(handler_error)
    logger.addHandler(handler_access)


UNIT_TEST_MODE = os.environ.get('UNIT_TEST_MODE', False)

"""
WARNING! ORDER IS IMPORTANT!!

"""

"""
REGISTER BLUEPRINT
"""

for blueprint in get_registered_blueprint():
    flask_app.register_blueprint(blueprint)


"""
SETUP LOGGING
"""

if not UNIT_TEST_MODE:
    setup_logging()


"""
ERROR HANDLER
"""
