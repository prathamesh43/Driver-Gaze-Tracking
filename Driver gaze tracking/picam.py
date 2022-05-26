
import cv2
import time
from picamera.array import PiRGBArray
from picamera import PiCamera

# define a video capture object
camera = PiCamera()

# Set the camera resolution
camera.resolution = (1280, 720)

# Set the number of frames per second
camera.framerate = 30

# Generates a 3D RGB array and stores it in rawCapture
raw_capture = PiRGBArray(camera, size=(1280, 720))

# Wait a certain number of seconds to allow the camera time to warmup
time.sleep(0.5)
size = (1280, 720)
font = cv2.FONT_HERSHEY_SIMPLEX
start_time = time.time()
fps = 0
while (True):

    # Capture the video frame
    # by frame
    try:
        frame = camera.capture_continuous(raw_capture, format="bgr", use_video_port=True)
        img = frame.array
        cv2.imshow("img", img)
    except:
        print("error")

camera.release()
# Destroy all the windows
cv2.destroyAllWindows()