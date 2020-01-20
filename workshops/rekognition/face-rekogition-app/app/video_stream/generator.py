from app.video_stream.camera import Camera


def encode_and_generate_video():
    camera = Camera()

    while True:
        frame = camera.get_frame()

        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n'
              + bytearray(frame) + b'\r\n')
