import pandas as pd
import time
import utils
from face_detector import get_face_detector, find_faces
import matplotlib.pyplot as plt
from headpose import *
import cv2
import numpy as np

face_model = get_face_detector()
fps = 0
ar = 0
start_S = False

y, p, r = 0, 0, 0
avg_yaw, avg_pitch, avg_roll = [], [], []
yaw_angles = []
pitch_angles = []
roll_angles = []

hght = []
cntr = []
area = []
fpsl = []

ryaw_angles = []
rpitch_angles = []
rroll_angles = []

# img = np.zeros((885, 1280, 3), np.uint8)
# define a video capture object
fname = "r4"

vid = cv2.VideoCapture("New/vids/" + fname + ".mp4")
frame_width = int(vid.get(4))
frame_height = int(vid.get(3))
sze = (frame_width, frame_height)
#

imgs = []
svid = cv2.VideoWriter("New/vids/vids_p/" + fname + "_p.avi", cv2.VideoWriter_fourcc(*'MJPG'), 30, sze)

# for x in range(1,14):
#     if x < 10:
#         timg = cv2.imread("imgs/0"+str(x)+".jpg")
#         imgs.append(timg)
#     else:
#         timg = cv2.imread("imgs/" + str(x)+".jpg")
#         imgs.append(timg)

hb = False
tempht = 0
tempcntr = 0
start_time = time.time()

while True:

    # Capture the video frame by frame
    ret, img = vid.read()
    if ret == True:
        img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
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

        print(str(fps) + ' %f %f %f\n' % (y, p, r))
        utils.draw_axis(img2, y, p, r)
        cv2.putText(img2, ("Pitch:" + str(p)), tuple((30, 210)), cv2.FONT_HERSHEY_PLAIN, 2, (255, 2, 2), 2)
        cv2.putText(img2, ("Yaw:" + str(y)), tuple((30, 160)), cv2.FONT_HERSHEY_PLAIN, 2, (2, 255, 2), 2)
        cv2.putText(img2, ("Frame:" + str(fps)), tuple((30, 260)), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        #cv2.imshow("img", img2)

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

        yaw_angles.append(y)
        pitch_angles.append(p)
        roll_angles.append(r)
        area.append(ar)
        hght.append(tempht)
        cntr.append(tempcntr)

        fps += 1
        fpsl.append(fps)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            end_time = time.time()
            break
        # time.sleep(0.032)

    else:
        break

end_time = time.time()
vid.release()
svid.release()
# Destroy all the windows
cv2.destroyAllWindows()
print(fps / (end_time - start_time))

data_dict = {"fps": fpsl, "yaw": yaw_angles, "pitch": pitch_angles, "roll": roll_angles, "centre": cntr, "area": area,
             "height": hght}
df = pd.DataFrame(data_dict)
# df = df.round(2)

df.to_csv("New/vids/vids_p/" + fname + '_f.csv', header=True, index=False)
# print(df)


plt.figure()
plt.subplot(411)
plt.plot(yaw_angles)
plt.plot(color="blue")
# plt.xlabel('Frame No.')
plt.ylabel('Yaw angle (deg)')
plt.xticks(np.arange(0, fps + 1, 300), )

plt.subplot(412)
plt.plot(pitch_angles)
plt.plot(color="red")
# plt.xlabel('Frame No.')
plt.ylabel('Pitch angle (deg)')
plt.xticks(np.arange(0, fps + 1, 300))

plt.subplot(413)
plt.plot(roll_angles)
plt.plot(color="green")
# plt.xlabel('Frame No.')
plt.ylabel('Roll angle (deg)')
plt.xticks(np.arange(0, fps + 1, 300))
#
plt.subplot(414)
plt.plot(yaw_angles, label='Yaw')
plt.plot(pitch_angles, label='Pitch')
plt.plot(roll_angles, label='Roll')
plt.xticks(np.arange(0, fps + 1, 300))
plt.xlabel('Frame No.')
plt.ylabel('Angle (deg)')
plt.legend()
plt.savefig("New/vids/vids_p/" + fname + "_plot", format="jpg")
plt.show()

#
# print(yaw_angles, pitch_angles, roll_angle
