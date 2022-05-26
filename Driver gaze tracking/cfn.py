
import torch
import cv2
import stable_hopenetlite
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import time
import utils
from picamera.array import PiRGBArray
from picamera import PiCamera
from face_detector import get_face_detector, find_faces


transformations = transforms.Compose([transforms.Scale(224),
                                      transforms.CenterCrop(224), transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

pos_net = stable_hopenetlite.shufflenet_v2_x1_0()
saved_state_dict = torch.load('model/shuff_epoch_120.pkl', map_location="cpu")
pos_net.load_state_dict(saved_state_dict, strict=False)
pos_net.eval()

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
face_model = get_face_detector()
start_time = time.time()
fps = 0
while (True):

    # Capture the video frame
    # by frame
    try:
        frame = camera.capture_continuous(raw_capture, format="bgr", use_video_port=True)
        img = frame.array
        img2 = img.copy()
    except:
        img = cv2.imread("headpose.png")
    try:
        faces = find_faces(img, face_model)
        for x in faces:
            img = img[x[1]-20:x[3]+20,x[0]-20:x[2]+20]
            img2 = img.copy()
            break
    except:
        img2 = img.copy()
        pass

    img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(img)

    # Transform
    img = transformations(img)
    img_shape = img.size()
    img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
    img = Variable(img)

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor)


    yaw, pitch, roll = pos_net(img)
    #print(x)

    yaw_predicted = F.softmax(yaw)
    pitch_predicted = F.softmax(pitch)
    roll_predicted = F.softmax(roll)
    # Get continuous predictions in degrees.
    yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
    pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
    roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

    print(str(fps) + ' %f %f %f\n' % (yaw_predicted, pitch_predicted, roll_predicted))
    utils.draw_axis(img2, yaw_predicted, pitch_predicted, roll_predicted)

    cv2.imshow('frame', img2)
    #cv2.moveWindow('frame', 0, 0)
    fps += 1
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        end_time = time.time()
        break

print(fps/(end_time-start_time))
camera.release()
# Destroy all the windows
cv2.destroyAllWindows()