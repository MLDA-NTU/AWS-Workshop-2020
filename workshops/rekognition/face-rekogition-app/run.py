import os
from app import flask_app
# from app.video_stream import stop_capture


if __name__ == '__main__':
    flask_app.run(
        host=os.environ.get('HOST', '127.0.0.1'),
        port=os.environ.get('PORT', 5000),
        debug=os.environ.get('DEBUG', False)
    )
