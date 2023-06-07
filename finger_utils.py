# Finger landmark IDs
WRIST = 0

THUMB_BOTTOM = 1
THUMB_BOTTOM_MID = 2
THUMB_TOP_MID = 3
THUMB_TIP = 4

INDEX_BOTTOM = 5
INDEX_BOTTOM_MID = 6
INDEX_TOP_MID = 7
INDEX_TIP = 8

MIDDLE_BOTTOM = 9
MIDDLE_BOTTOM_MID = 10
MIDDLE_TOP_MID = 11
MIDDLE_TIP = 12

RING_BOTTOM = 13
RING_BOTTOM_MID = 14
RING_TOP_MID = 15
RING_TIP = 16

PINKY_BOTTOM = 17
PINKY_BOTTOM_MID = 18
PINKY_TOP_MID = 19
PINKY_TIP = 20

# Check if the hand is upside down
def hand_down(points):
    return points[MIDDLE_TIP].y > points[WRIST].y

def palm_up(points):
    return points[THUMB_TIP].y < points[THUMB_TOP_MID].y \
        and points[INDEX_TIP].y < points[INDEX_TOP_MID].y \
        and points[MIDDLE_TIP].y < points[MIDDLE_TOP_MID].y \
        and points[RING_TIP].y < points[RING_TOP_MID].y \
        and points[PINKY_TIP].y < points[PINKY_TOP_MID].y