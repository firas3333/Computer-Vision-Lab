import numpy as np
import cv2 as cv

inputVideoName = 'highway.avi'
selectPoints = True
numberOfPoints = 50

cap = cv.VideoCapture(inputVideoName)

first_point_selected = False
points_selected = np.empty([0, 1, 2], dtype=np.float32)

min_distance = 10
# params for ShiTomasi corner detection
# Its based on Harris Corner detector but with additional improvements
feature_params = dict(maxCorners=numberOfPoints,
                      qualityLevel=0.1,
                      minDistance=min_distance,
                      blockSize=10)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(10, 10),
                 maxLevel=4,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))


# Mouse function
def select_point(event, x, y, flags, params):
    global points_selected, first_point_selected
    if event == cv.EVENT_LBUTTONDOWN:
        user_points = np.empty([1, 1, 2], dtype=np.float32)
        user_points[0][0] = [x, y]
        points_selected = np.concatenate([points_selected, user_points])
        cv.circle(old_frame, (x, y), 5, (0, 255, 0), -1)


# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

if selectPoints:
    print('Please Select points for features tracking')
    print('Click on features in "Select Points" img and then click ESC when you finish')
    cv.namedWindow('Select Points')
    cv.setMouseCallback('Select Points', select_point)
    while True:
        cv.imshow('Select Points', old_frame)
        k = cv.waitKey(20)
        # escape
        if k == 27:
            break
else:
    points_selected = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)


def getFreaturesMask():
    features_mask = np.zeros(old_frame.shape[:2], np.uint8)
    rows, cols = features_mask.shape
    for point in points_selected:
        x, y = point.ravel()
        fromRow = max(int(x - min_distance), 0)
        toRow = min(int(x + min_distance), rows)
        fromCol = max(int(y - min_distance), 0)
        toCol = min(int(y + min_distance), cols)
        features_mask[fromRow:toRow, fromCol:toCol] = 255
    return features_mask


# Create a mask image filled with zeros for drawing purposes
mask = np.zeros_like(old_frame)

global img

while True:
    ret, frame = cap.read()
    if frame is not None:
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # if selectPoints is False:
        # points_selected = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        # points_selected = np.concatenate([points_selected, new_points_selected])

        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, points_selected, None, **lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = points_selected[st == 1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(p1, points_selected)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (a, b), (c, d), (0, 0, 255), 1)
        img = cv.add(frame, mask)
        cv.imshow('Frame', img)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

        old_gray = frame_gray.copy()

        # If some features points ended or stopped to be tracked
        # Search for new features points to be tracked
        if selectPoints is False and len(p1) < numberOfPoints:
            new_features_mask = getFreaturesMask()
            new_points = cv.goodFeaturesToTrack(old_gray, mask=new_features_mask,
                                                maxCorners=numberOfPoints - len(p1),
                                                qualityLevel=0.01,
                                                minDistance=min_distance,
                                                blockSize=7)
            points_selected = np.concatenate([points_selected, new_points])
        else:
            points_selected = good_new.reshape(-1, 1, 2)

    else:
        cap.release()
        break

cv.imwrite('image_optical_flow.jpg', img)
cv.destroyAllWindows()
