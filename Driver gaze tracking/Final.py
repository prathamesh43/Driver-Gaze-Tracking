import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
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
from tensorflow.keras.models import load_model
from pickle import load


def det_eor(y, p, r, h, a):
    global model, scaler, state

    st = False
    df = pd.DataFrame({"y": [y], "p": [p], "r": [r], "h": [h], 'a': [a]})
    print(df)
    X_test = df.iloc[:, :]
    # X_test = [[y,p,r,h,a]]
    feat = scaler.transform(X_test)
    print(feat)
    p = model.predict(feat)
    if p[0][1].round(2) > 0.6 :
        st = True

    return st


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




model = load_model('models/0.9733351469039917.h5')
model.summary()
scaler = load(open('scaler.pkl', 'rb'))
print(scaler)

# print(det_eor(0.02,15.85,-5.98,110,39711))

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
fpsl2 = []

ryaw_angles = []
rpitch_angles = []
rroll_angles = []

fname = "j2"

vid = cv2.VideoCapture("vids/" + fname + ".mp4")
frame_width = int(vid.get(3))
frame_height = int(vid.get(4))
sze = (frame_width, frame_height)
#

imgs = []
svid = cv2.VideoWriter("vids/" + fname + "_proc.avi", cv2.VideoWriter_fourcc(*'MJPG'), 30, sze)


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

        yaw_predicted = F.softmax(yaw, dim=yaw.shape[0])
        pitch_predicted = F.softmax(pitch, dim=pitch.shape[0])
        roll_predicted = F.softmax(roll, dim=roll.shape[0])
        # Get continuous predictions in degrees.
        yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
        roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

        y0, p0, r0 = float(yaw_predicted), float(pitch_predicted), float(roll_predicted)


        y = round(get_avg_y(y0), 2)
        p = round(get_avg_p(p0), 2)
        r = round(get_avg_r(r0), 2)

        fps += 1
        fpsl.append(fps)

        print(str(fps) + ' %f %f %f\n' % (y, p, r))
        # utils.draw_axis(img2, y, p, r)
        # cv2.putText(img2, ("Pitch:" + str(p)), tuple((10, 20)), cv2.FONT_HERSHEY_PLAIN, 1.5, (1, 200, 255), 1)
        # cv2.putText(img2, ("Yaw:" + str(y)), tuple((10, 50)), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 20, 255), 1)

        state = False
        # stt1 = time.time()
        #
        if fps % 3 == 0:
            state = det_eor(y,p,r,tempht,ar)


            if state:
                cv2.putText(img2, "All Good !!", tuple((550, 650)), cv2.FONT_HERSHEY_PLAIN, 1.5, (20, 255, 2), 2)

            else:
                cv2.putText(img2, "EOR !!!!!!!!!", tuple((500, 650)), cv2.FONT_HERSHEY_PLAIN, 2.7, (1, 2, 255), 2)
        #
        # stt2 = time.time()
        # fpsl2.append(stt2 - stt1)

        cv2.imshow('frame', img2)
        svid.write(img2)
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

        # yaw_angles.append(y)
        # pitch_angles.append(p)
        # roll_angles.append(r)
        # area.append(ar)
        # hght.append(tempht)




        if cv2.waitKey(1) & 0xFF == ord('q'):
            end_time = time.time()
            break
        #time.sleep(0.032)

    else:
        break

end_time = time.time()
vid.release()
svid.release()
# Destroy all the windows
cv2.destroyAllWindows()
print(fps / (end_time - start_time))
# print(sum(fpsl2)/len(fpsl2))