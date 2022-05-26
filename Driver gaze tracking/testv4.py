import pandas as pd
import cv2
import time
import utils
from face_detector import get_face_detector, find_faces
from keras.models import load_model
from pickle import load
from headpose import *
import numpy as np


def det_eor(y, p, r, c, a, h):
    global model, scaler, state

    st = False
    df = pd.DataFrame({"y": [y], "p": [p], "r": [r], 'c': [c], 'a': [a], "h": [h]})
    # print(df)
    X_test = df.iloc[:, :]
    # X_test = [[y,p,r,h,a]]
    feat = scaler.transform(X_test)
    # print(feat)
    p = model.predict(feat)
    if p[0][1].round(2) > 0.7:
        st = True

    return st


def war_sys():
    pass


model = load_model('models/0.9990435242652893.h5')
model.summary()
scaler = load(open('scaler_prj.pkl', 'rb'))
print(scaler)

pos_net = stable_hopenetlite.shufflenet_v2_x1_0()
saved_state_dict = torch.load('model/shuff_epoch_120.pkl', map_location="cpu")
pos_net.load_state_dict(saved_state_dict, strict=False)
pos_net.eval()

face_model = get_face_detector()

sp_df = pd.read_csv("FUDS.csv")
spend = sp_df.shape
print(spend)

fps = 0
ar = 0
y, p, r = 0, 0, 0
e_on_st, e_off_st = 0, 0
start_S = False

avg_yaw, avg_pitch, avg_roll = [0], [0], [0]
yaw_angles = [0]
pitch_angles = [0]
roll_angles = [0]

fname = "r4"
# vid = cv2.VideoCapture(0)
vid = cv2.VideoCapture("New/vids/" + fname + ".mp4")

frame_width = int(vid.get(4))
frame_height = int(vid.get(3))
sze = (frame_width, frame_height)
print(sze)

imgs = []
svid = cv2.VideoWriter("New/final/" + fname + "_p2.avi", cv2.VideoWriter_fourcc(*'MJPG'), 25, (1920, 1080))

hb = False
tempht = 0
tempcntr = 0
state = False
pstate = False
on_cnt, off_cnt = 0, 0
i = 1
dt = 30
at = 20
no_war = 0
war_state = False
wart = 0

start_time = time.time()

while True:

    ret, img = vid.read()
    if ret:
        img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
        # img = img[20:701, ]
        # img = cv2.resize(img, (450,800))
        img2 = img.copy()
        try:
            faces = find_faces(img, face_model)
            for x in faces:
                img = img[x[1]:x[3], x[0]:x[2]]
                # img3 = img.copy()
                # img3 = img3[x[1] - 20:x[3] + 20, x[0] +30 :x[2] - 30]
                # img = img[x[1] - 20:x[3] + 20, x[0] - 20:x[2] + 20]
                # img2 = img.copy()
                tempht = (x[1])
                tempcntr = ((x[1] + x[3]) / 2)
                ar = (x[3] - x[1]) * (x[2] - x[0])
                # hb = True
                break

            try:
                y, p, r = get_hp(img)
            except:
                y, p, r = yaw_angles[-1], pitch_angles[-1], roll_angles[-1]

        except:
            img2 = img.copy()

        fps += 1
        print(fps)

        # print(str(fps) + ' %f %f %f\n' % (y, p, r))
        utils.draw_axis(img2, y, p, r)
        # cv2.putText(img2, ("Pitch:" + str(p)), tuple((10, 20)), cv2.FONT_HERSHEY_PLAIN, 1.5, (1, 200, 255), 1)
        # cv2.putText(img2, ("Yaw:" + str(y)), tuple((10, 50)), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 20, 255), 1)

        # stt1 = time.time()
        #
        # if fps % 2 == 0:
        state = det_eor(y, p, r, tempcntr, ar, tempht)

        if state:
            cv2.putText(img2, "All Good !!", tuple((240, 65)), cv2.FONT_HERSHEY_PLAIN, 2.2, (20, 255, 2), 2)
            wart = 100
        else:
            cv2.putText(img2, "EOR !!!!!!!", tuple((240, 65)), cv2.FONT_HERSHEY_PLAIN, 2.7, (1, 2, 255), 2)

        img2 = cv2.resize(img2, (608, 1080))

        img4 = np.zeros((1080, 1920, 3), np.uint8)
        img4[:, :] = 255
        img4 = cv2.cvtColor(img4, cv2.COLOR_RGB2BGR)
        img4[0:1080, 656:1264, :] = img2

        cspeed = round(sp_df.iloc[i, 0], 2)

        if cspeed < 20:
            dt = 100
            at = 15
        elif 20 < cspeed < 60:
            dt = 75
            at = 20
        elif 60 < cspeed < 100:
            dt = 60
            at = 30
        elif cspeed >= 100:
            dt = 45
            at = 30

        if state:
            on_cnt += 1
            if on_cnt >= at:
                war_state = False
                off_cnt = 0
        else:
            off_cnt += 1
            if off_cnt >= dt:
                on_cnt = 0
                war_state = True
                cv2.putText(img4, "Warning!!!", tuple((760, 350)), cv2.FONT_HERSHEY_SIMPLEX, 2.8, (1, 2, 240), 2)

        if war_state and dt == off_cnt:
            no_war += 1

        if (fps % 20) == 0:
            if i == 1370:
                i = 10
            else:
                i += 1

        cv2.putText(img4, "Eyes On Road Time: " + str(round(on_cnt * 0.0334, 1)), tuple((50, 60)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (1, 2, 255), 2)
        cv2.putText(img4, "Eyes Off Road Time: " + str(round(off_cnt * 0.0334, 1)), tuple((50, 120)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (1, 2, 255), 2)
        cv2.putText(img4, "Vehicle Speed: " + str(cspeed) + " kmph", tuple((50, 180)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (1, 2, 255), 2)
        cv2.putText(img4, "No. of Warnings: " + str(no_war), tuple((50, 240)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (1, 2, 255), 2)

        # cv2.imshow("out", img4)
        svid.write(img4)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            end_time = time.time()
            break

    else:
        break

end_time = time.time()
vid.release()
svid.release()
# Destroy all the windows
cv2.destroyAllWindows()
print(fps / (end_time - start_time))
# print(sum(fpsl2)/len(fpsl2))
