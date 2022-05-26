import torch
import stable_hopenetlite
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F



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


avg_yaw, avg_pitch, avg_roll = [], [], []

transformations = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224), transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

pos_net = stable_hopenetlite.shufflenet_v2_x1_0()
saved_state_dict = torch.load('model/shuff_epoch_120.pkl', map_location="cpu")
pos_net.load_state_dict(saved_state_dict, strict=False)
pos_net.eval()


def get_hp(img):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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

    yaw_predicted = F.softmax(yaw, dim=1)
    pitch_predicted = F.softmax(pitch, dim=1)
    roll_predicted = F.softmax(roll, dim=1)
    # Get continuous predictions in degrees.
    yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
    pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
    roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

    y0, p0, r0 = float(yaw_predicted), float(pitch_predicted), float(roll_predicted)

    y = round(get_avg_y(y0), 2)
    p = round(get_avg_p(p0), 2)
    r = round(get_avg_r(r0), 2)

    return y,p,r