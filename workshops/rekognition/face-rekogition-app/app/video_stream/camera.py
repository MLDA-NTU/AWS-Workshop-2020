import os
import time
import threading

import cv2


class CameraEvent:
    """An event driven class which scope is to signal active clients
    when new frame is available
    """

    def __init__(self):
        self.events = {}

    def wait(self):
        """Invoke from each client's thread to wait for the next frame
        """
        thread_id = threading.get_ident()
        if not (thread_id in self.events):
            # add new clients
            self.events[thread_id] = [threading.Event(), time.time()]
        return self.events[thread_id][0].wait()

    def set_event(self):
        """Invoked by cmera thread when new frame is available
        """
        now = time.time()
        remove = None

        for thread_id, event in self.events.items():
            if not event[0].isSet():
                event[0].set()
                event[1] = now
            else:
                if now - event[1] > 5:
                    remove = thread_id
        if remove:
            del self.events[remove]

    def clear(self):
        """Invoke from each client's thread after a frame is processed
        """
        self.events[threading.get_ident()][0].clear()


class BaseCamera:
    """Abstract class for event-based camera streamer
    """
    __abstract__ = True

    _thread = None
    _frame = None
    _last_access = 0
    _event = CameraEvent()

    def __init__(self):
        """Start the background camera thread
        """
        if BaseCamera._thread is None:
            BaseCamera._last_access = time.time()

            BaseCamera._thread = threading.Thread(target=self.thread)
            BaseCamera._thread.start()

            while self.get_frame() is None:
                time.sleep(0)

    def get_frame(self):
        """Get the current frame from camera
        """
        BaseCamera._last_access = time.time()

        BaseCamera._event.wait()
        BaseCamera._event.clear()

        return BaseCamera._frame

    @staticmethod
    def generate_frames():
        """Generator that stream frames from camera
        """
        raise NotImplementedError('Not implemented yet: function %s' % BaseCamera.generate_frames)

    @classmethod
    def thread(cls):
        """Background thread sending signal or stopping thread
        when client is inactive
        """
        print('Starting camera thread')

        for frame in cls.generate_frames():
            BaseCamera._frame = frame
            BaseCamera._event.set_event()  # send signal to clients
            time.sleep(0)

            if time.time() - BaseCamera._last_access > 10:
                cls.generate_frames().close()

                print('Stopping camera due to inactivity')
                break

        BaseCamera._thread = None


class Camera(BaseCamera):
    video_source = 0

    def __init__(self):
        Camera.video_source = int(os.environ.get('OPENCV_CAMERA_SOURCE', 0))

        super().__init__()

    @staticmethod
    def generate_frames():
        camera = cv2.VideoCapture(Camera.video_source)

        if not camera.isOpened():
            raise RuntimeError('Unable to open the camera')

        while True:
            success, frame = camera.read()

            if not success:
                continue

            yield cv2.imencode('.jpg', frame)[1].tobytes()
