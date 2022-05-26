import pandas as pd
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

face_model = get_face_detector()
fps = 0
ar = 0
start_S = False

avg_yaw, avg_pitch, avg_roll = [], [], []
yaw_angles = []
pitch_angles = []
roll_angles = []

hght = []
area = []
fpsl = []

ryaw_angles = []
rpitch_angles = []
rroll_angles = []

# img = np.zeros((885, 1280, 3), np.uint8)
# define a video capture object
fname = "s"

vid = cv2.VideoCapture("vids/" + fname + ".mp4")
frame_width = int(vid.get(3))
frame_height = int(vid.get(4))
sze = (frame_width, frame_height)
#

imgs = []
# svid = cv2.VideoWriter("vids/" + fname + "_p.avi", cv2.VideoWriter_fourcc(*'MJPG'), 30, sze)


# for x in range(1,14):
#     if x < 10:
#         timg = cv2.imread("imgs/0"+str(x)+".jpg")
#         imgs.append(timg)
#     else:
#         timg = cv2.imread("imgs/" + str(x)+".jpg")
#         imgs.append(timg)

hb = False
tempht = 0
start_time = time.time()
while True:

    # Capture the video frame
    # by frame
    ret, img = vid.read()
    if ret == True:
        img2 = img.copy()
        try:
            faces = find_faces(img, face_model)
            for x in faces:
                img3 = img.copy()
                img3 = img3[x[1] - 20:x[3] + 20, x[0] +30 :x[2] - 30]
                img = img[x[1] - 20:x[3] + 20, x[0] - 20:x[2] + 20]
                #img2 = img.copy()
                tempht = (x[1])
                ar = (x[3]-x[1])*(x[2]-x[0])
                # hb = True

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

        y0, p0, r0 = float(yaw_predicted), float(pitch_predicted), float(roll_predicted)

        y = round(get_avg_y(y0), 2)
        p = round(get_avg_p(p0), 2)
        r = round(get_avg_r(r0), 2)

        print(str(fps) + ' %f %f %f\n' % (y, p, r))
        utils.draw_axis(img2, y, p, r)
        cv2.putText(img2, ("Pitch:" + str(p)), tuple((10, 20)), cv2.FONT_HERSHEY_PLAIN, 1.5, (128, 255, 255), 1)
        cv2.putText(img2, ("Yaw:" + str(y)), tuple((10, 50)), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 128), 1)

        cv2.imshow('frame', img2)
        # svid.write(img2)
        # cv2.moveWindow('frame', 700, 0)
        # test_img = test_im.copy()
        # cnt = get_pnt(y, p, r)
        # print(cnt)
        # cv2.circle(test_img, cnt, 20, (2, 5, 255))
        # font = cv2.FONT_HERSHEY_PLAIN
        # # cv2.putText(test_img, str(p), p, font, 1.5, (0, 0, 255))
        # #test_img = cv2.resize(test_img,(640,442))
        #

        # cv2.imshow("testimg", test_img)
        # cv2.moveWindow("testimg", 500, 0)

        yaw_angles.append(y)
        pitch_angles.append(p)
        roll_angles.append(r)
        area.append(ar)
        hght.append(tempht)

        ryaw_angles.append(y0)
        rpitch_angles.append(p0)
        rroll_angles.append(r0)

        fps += 1
        fpsl.append(fps)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            end_time = time.time()
            break
        #time.sleep(0.032)

    else:
        break

end_time = time.time()
vid.release()
#svid.release()
# Destroy all the windows
cv2.destroyAllWindows()
print(fps / (end_time - start_time))

data_dict = {"fps":fpsl, "yaw":yaw_angles, "pitch":pitch_angles, "roll":roll_angles, "height":hght, "area": area}
df = pd.DataFrame(data_dict)
# df = df.round(2)
# noinspection PyTypeChecker
df.to_csv("vids/" + fname +'_f.csv', header=True, index=False)
print(df)


# plt.figure()
# plt.subplot(411)
# plt.plot(yaw_angles)
# plt.plot(ryaw_angles)
#
# plt.subplot(412)
# plt.plot(pitch_angles)
# plt.plot(rpitch_angles)
#
# plt.subplot(413)
# plt.plot(roll_angles)
# plt.plot(rroll_angles)
# #
# plt.subplot(414)
# plt.plot(ryaw_angles)
# plt.plot(rpitch_angles)
# plt.plot(rroll_angles)
# plt.show()

#plt.savefig("vids/"+fname+'_plot.png')
#
# print(yaw_angles, pitch_angles, roll_angle

