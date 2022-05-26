import torch
import cv2
import stable_hopenetlite
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import time
import utils
from face_detector import get_face_detector, find_faces
import numpy as np
import matplotlib.pyplot as plt


def get_avg_y(agl):
    global avg_yaw

    avg_yaw.append(agl)
    if len(avg_yaw) <= 4:

        return agl

    elif len(avg_yaw) == 5:
        mean = sum(avg_yaw)/len(avg_yaw)
        avg_yaw.pop(0)
        return mean
    else:

        return agl


def get_avg_p(agl):
    global avg_pitch

    avg_pitch.append(agl)
    if len(avg_pitch) <= 4:

        return agl

    elif len(avg_pitch) == 5:
        mean = sum(avg_pitch) / len(avg_pitch)
        avg_pitch.pop(0)
        return mean
    else:
        return agl


def get_avg_r(agl):
    global avg_roll

    avg_roll.append(agl)
    if len(avg_roll) <= 4:
        return agl
    elif len(avg_roll) == 5:
        mean = sum(avg_roll) / len(avg_roll)
        avg_roll.pop(0)
        return mean
    else:
        return agl


def get_pnt(y, p, r):
    x1 = 11.6 * (y + 45) + 115
    x2 = (-14.6*p) + 350
    pnt = (int(x1), int(x2))
    return pnt


transformations = transforms.Compose([transforms.Scale(224),
                                      transforms.CenterCrop(224), transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

pos_net = stable_hopenetlite.shufflenet_v2_x1_0()
saved_state_dict = torch.load('model/shuff_epoch_120.pkl', map_location="cpu")
pos_net.load_state_dict(saved_state_dict, strict=False)
pos_net.eval()

vid = cv2.VideoCapture(0)
face_model = get_face_detector()
fps = 0
start_S = False

avg_yaw, avg_pitch, avg_roll = [], [], []
yaw_angles = []
pitch_angles = []
roll_angles = []

img = np.zeros((885, 1280, 3), np.uint8)
# define a video capture object
# vid = cv2.VideoCapture(1)
test_im = cv2.imread("FR2.jpg")
imgs = []

for x in range(1,14):
    if x < 10:
        timg = cv2.imread("imgs/0"+str(x)+".jpg")
        imgs.append(timg)
    else:
        timg = cv2.imread("imgs/" + str(x)+".jpg")
        imgs.append(timg)


start_time = time.time()
for x in imgs:

    # Capture the video frame
    # by frame
    ret, img = True, x
    if ret == True:
        img2 = img.copy()
        try:
            faces = find_faces(img, face_model)
            for x in faces:
                img = img[x[1] - 20:x[3] + 20, x[0] - 20:x[2] + 20]
                img2 = img.copy()

                break
        except:
            img2 = img.copy()
            pass

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(img)

        # Transform
        img = transformations(img)
        img_shape = img.size()
        img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
        img = Variable(img)

        idx_tensor = [idx for idx in range(66)]
        idx_tensor = torch.FloatTensor(idx_tensor)

        yaw, pitch, roll = pos_net(img)
        # print(x)

        yaw_predicted = F.softmax(yaw)
        pitch_predicted = F.softmax(pitch)
        roll_predicted = F.softmax(roll)
        # Get continuous predictions in degrees.
        yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
        roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

        y0, p, r = float(yaw_predicted), float(pitch_predicted), float(roll_predicted)

        y= get_avg_y(y0)
        p = get_avg_p(p)
        r = get_avg_r(r)

        print(str(fps) + ' %f %f %f\n' % (y, p, r))
        utils.draw_axis(img2, y, p, r)
        cv2.putText(img2, ("Pitch:" + str(p)), tuple((10, 20)), cv2.FONT_HERSHEY_PLAIN, 1.5, (128, 255, 255), 1)
        cv2.putText(img2, ("Yaw:" + str(y)), tuple((10, 50)), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 128), 1)

        cv2.imshow('frame', img2)
        # cv2.moveWindow('frame', 700, 0)
        test_img = test_im.copy()
        cnt = get_pnt(y, p, r)
        print(cnt)
        cv2.circle(test_img, cnt, 20, (2, 5, 255))
        font = cv2.FONT_HERSHEY_PLAIN
        # cv2.putText(test_img, str(p), p, font, 1.5, (0, 0, 255))
        #test_img = cv2.resize(test_img,(640,442))

        cv2.imshow("testimg", test_img)
        cv2.moveWindow("testimg", 500, 0)
        cv2.imwrite("imgs/"+str(fps)+"_p.jpg", img2)

        fps += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            end_time = time.time()
            break
        #time.sleep(0.032)

    else:
        pass

end_time = time.time()
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
print(fps / (end_time - start_time))
# plt.figure()
# plt.subplot(411)
# plt.plot(yaw_angles)
#
# plt.subplot(412)
# plt.plot(yaw_angles2)
#
# plt.subplot(413)
# plt.plot(yaw_angles3)
#
# plt.subplot(414)
# plt.plot(raw_yaw)
# plt.show()
#
# print(yaw_angles, pitch_angles, roll_angles)


