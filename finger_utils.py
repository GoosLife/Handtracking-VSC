# Check if the hand is upside down
def hand_down(points):
    return points[MIDDLE_FINGER_TIP].y > points[WRIST].y