from flask import (
    Blueprint, Response, render_template
)
from app.video_stream.generator import encode_and_generate_video

mod_video_stream = Blueprint('videostream', __name__, url_prefix='/videostream/')


@mod_video_stream.route('/')
def web_client_view():
    return render_template('index.html')


@mod_video_stream.route('/videofeed')
def api_video_view():
    # stream video from webcam with preprocessed information
    return Response(
        encode_and_generate_video(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )
