def get_avg_y(agl):
    global avg_yaw

    avg_yaw.append(agl)
    if len(avg_yaw) <= 2:

        return agl

    elif len(avg_yaw) == 3:
        mean = sum(avg_yaw) / len(avg_yaw)
        print("mean:{}".format(mean) + "Angle:{}".format(agl))
        if abs(avg_yaw[-1] - mean) > mean * 0.7:
            avg_yaw.pop(-1)
            mean = sum(avg_yaw) / len(avg_yaw)
            return mean
        else:
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
        mean = sum(avg_pitch) / 4
        if abs(avg_pitch[3] - mean) > mean * 0.7:
            avg_pitch.pop(3)
            mean = sum(avg_pitch) / 3
            return mean
        else:
            avg_pitch.pop(0)
            return mean

    else:
        return agl


def get_avg_r(agl):
    global avg_roll

    avg_roll.append(agl)
    if len(avg_roll) <= 3:

        return agl

    elif len(avg_roll) == 4:
        mean = sum(avg_roll) / 4
        if abs(avg_roll[3] - mean) > mean * 0.7:
            avg_roll.pop(3)
            mean = sum(avg_roll) / 3
            return mean
        else:
            avg_roll.pop(0)
            return mean

    else:
        return agl
