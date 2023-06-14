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

ID = 0
X = 1
Y = 2

def hand_down(points):
    for finger_id in range(1, 21):
        if points[finger_id][Y] <= points[WRIST][Y]:
            return False
    return True

def palm_up(points: list):
    return points[THUMB_TIP][Y] < points[THUMB_BOTTOM][Y] \
        and points[INDEX_TIP][Y] < points[INDEX_TOP_MID][Y] \
        and points[MIDDLE_TIP][Y] < points[MIDDLE_TOP_MID][Y] \
        and points[RING_TIP][Y] < points[RING_TOP_MID][Y] \
        and points[PINKY_TIP][Y] < points[PINKY_TOP_MID][Y]
    
def pointing_left(points: list):
    # Check if the thumb is to the right of the index finger
    if points[THUMB_TIP][X] > points[INDEX_TIP][X]:
        # Check if the index finger tip is to the left of the rest of the index finger
        if points[INDEX_TIP][X] < points[INDEX_TOP_MID][X]:
            # Check if the middle, ring, and pinky fingers are pointing towards the palm
            if (
                points[MIDDLE_TIP][X] > points[MIDDLE_TOP_MID][X]
                and points[RING_TIP][X] > points[RING_TOP_MID][X]
                and points[PINKY_TIP][X] > points[PINKY_TOP_MID][X]
            ):
                return True
    return False

def pointing_right(points: list):
    # Check if the thumb is to the left of the index finger
    if points[THUMB_TIP][X] < points[INDEX_TIP][X]:
        # Check if the index finger tip is to the right of the rest of the index finger
        if points[INDEX_TIP][X] > points[INDEX_TOP_MID][X]:
            # Check if the middle, ring, and pinky fingers are pointing towards the palm
            if (
                points[MIDDLE_TIP][X] < points[MIDDLE_TOP_MID][X]
                and points[RING_TIP][X] < points[RING_TOP_MID][X]
                and points[PINKY_TIP][X] < points[PINKY_TOP_MID][X]
            ):
                return True
    return False