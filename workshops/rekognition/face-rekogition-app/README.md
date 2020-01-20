# Face Rekognition App

Face Recognition application which provide real-time face detection + recognition from video stream.
Using `OpenCV` and `AWS Rekognition API`

## Requirements

### Systems
- `python > 3.5 or 2.7`
- `virtualvenv or venv`
- `git`

### Python libraries
- `Flask 1.1`
- `OpenCV 3.4 (opencv-contrib-python)`
- `Numpy`

## Installation

```bash
virtualenv venv  # create local virtual environment
source virtualenv/bin/activate   # activate virtual environment
pip install -r requirements.txt  # install all python package required
```

## Run Server

```bash
export FLASK_APP=run.py      # define entry point
export FLASK_ENV=development # dev mode
export DEBUG=True            # dev mode
python run.py # available at 127.0.0.1:5000
```
