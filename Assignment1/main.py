# Interface for interactive selection of segement points

# Interface instruction:
# Image opens to the user once the program starts to run.
# The user then selects segments with mouse:
#	Right click: segments a point
#	Left click: start a line
#	Line is from the previous point selected to the one clicked.
#	All points in the line belong to the current segment
# User uses ג€˜space barג€™ to switch between segments, the message board shows on which segment the user is on at a given time.
# There are 4 segments in total, each has a color: RED, GREEN, BLUE, YELLOW respectively.
# Once you finish segmenting, press ג€˜Escג€™.
#
# Once manual segmentation is finished:
# The user will have four lists: seg0, seg1, seg2, seg3. Each is a list with all the points belonging to the segment.


# SHEREIN DABBAH-311382840
# FIRAS AYOUB-308185313

import cv2
import numpy as np

# use JPG images
inputImage = 'man.jpg'
segmentedImage=[]
segmaskImage =[]

SEGMENT_ZERO = 0
SEGMENT_ONE = 1
SEGMENT_TWO = 2
SEGMENT_THREE = 3

red_color = (0, 0, 255)
green_color = (0, 255, 0)
blue_color = (255, 0, 0)
yellow_color = (0, 255, 255)


# mouse callback function
def mouse_click(event, x, y, flags, params):
    # if left button is pressed, draw line
    if event == cv2.EVENT_LBUTTONDOWN:
        if current_segment == SEGMENT_ZERO:
            if len(seg0) == 0:
                seg0.append((x, y))
            else:
                points = add_line_point(seg0[-1], (x, y))
                seg0.extend(points)
        if current_segment == SEGMENT_ONE:
            if len(seg1) == 0:
                seg1.append((x, y))
            else:
                points = add_line_point(seg1[-1], (x, y))
                seg1.extend(points)
        if current_segment == SEGMENT_TWO:
            if len(seg2) == 0:
                seg2.append((x, y))
            else:
                points = add_line_point(seg2[-1], (x, y))
                seg2.extend(points)
        if current_segment == SEGMENT_THREE:
            if len(seg3) == 0:
                seg3.append((x, y))
            else:
                points = add_line_point(seg3[-1], (x, y))
                seg3.extend(points)

    # right mouse click adds single point
    if event == cv2.EVENT_RBUTTONDOWN:
        if current_segment == SEGMENT_ZERO:
            seg0.append((x, y))
        if current_segment == SEGMENT_ONE:
            seg1.append((x, y))
        if current_segment == SEGMENT_TWO:
            seg2.append((x, y))
        if current_segment == SEGMENT_THREE:
            seg3.append((x, y))

    # show on seg_img with colors
    paint_segment(seg0, (0, 0, 255))
    paint_segment(seg1, (0, 255, 0))
    paint_segment(seg2, (255, 0, 0))
    paint_segment(seg3, (0, 255, 255))


# given two points, this function returns all the points on line between.
# this is used when user selects lines on segments
def add_line_point(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    points = []
    issteep = abs(y2 - y1) > abs(x2 - x1)
    if issteep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    rev = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        rev = True
    deltax = x2 - x1
    deltay = abs(y2 - y1)
    error = int(deltax / 2)
    y = y1
    if y1 < y2:
        ystep = 1
    else:
        ystep = -1
    for x in range(x1, x2 + 1):
        if issteep:
            points.append((y, x))
        else:
            points.append((x, y))
        error -= deltay
        if error < 0:
            y += ystep
            error += deltax
    # Reverse the list if the coordinates were reversed
    if rev:
        points.reverse()
    return points


# given a segment points and a color, paint in seg_image
def paint_segment(segment, color):
    for center in segment:
        cv2.circle(seg_img, center, 2, color, -1)


# create seg mask(specify which areas are background, foreground)
# seg_fg is sure foreground pixels and the other segmentations arrays are sure background
# The rest of image pixels are probably BG
def create_mask(mask_shape, seg_fg, seg_bg1, seg_bg2, seg_bg3):
    # numpy.zeros -> Return a new array of given shape and type, filled with zeros.
    # create new mask with default value of probable foreground
    result_mask = np.full(mask_shape, cv2.GC_PR_BGD, np.uint8)

    # set sure foreground
    for col, row in seg_fg:
        result_mask[row][col] = cv2.GC_FGD

    # set sure background
    for col, row in seg_bg1:
        result_mask[row][col] = cv2.GC_BGD
    for col, row in seg_bg2:
        result_mask[row][col] = cv2.GC_BGD
    for col, row in seg_bg3:
        result_mask[row][col] = cv2.GC_BGD

    return result_mask


# we create for each segment an image of its color (only the pixels in the segment has color)
# then we add all the images to get the full colored img
def create_image_single_color_by_mask(mask, color):
    height, width = mask.shape[:2]
    result = np.zeros((height, width, 3), np.uint8)
    # set all pixels thats value!= 0 to same chosen color
    for i in range(height):
        for j in range(width):
            if mask[i][j] != 0:
                result[i][j] = color

    return result


def run_grab_cut_by_mask(image, mask):
    # bdgModel, fgdModel - These are arrays used by the grabCut algorithm
    # internally.two np.float64 type zero arrays of size (1,65).
    bgdm = np.zeros((1, 65), np.float64)
    fgdm = np.zeros((1, 65), np.float64)

    tmp_mask, res_bgd, res_fgd = cv2.grabCut(image, mask, None, bgdm, fgdm, 15, cv2.GC_INIT_WITH_MASK)

    # numpy.where -> Return elements chosen from x or y depending on condition.
    result_mask = np.where((tmp_mask == cv2.GC_BGD) | (tmp_mask == cv2.GC_PR_BGD), 0, 1).astype(np.uint8)
    return result_mask


def color_mask(mask, color):
    height, width = mask.shape[:2]
    result = np.zeros((height, width, 3), np.uint8)
    # set all pixels thats value!= 0 to same chosen color
    for i in range(height):
        for j in range(width):
            if mask[i][j] != 0:
                result[i][j] = color

    return result


def subtract_grab_cut(segment, mask, color1, color2):
    # segment 1
    bgdm = np.zeros((1, 65), np.float64)
    fgdm = np.zeros((1, 65), np.float64)

    # set mask for segment1 where BG and PR_BG are 0, PR_FG and FG are PR_BGD
    mask_segment = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), cv2.GC_BGD,
                            cv2.GC_PR_BGD).astype(np.uint8)

    # set mask to FGD in segment1
    for col, row in segment:
        mask_segment[row][col] = cv2.GC_FGD

    mask_segment, bgdm, fgdm = cv2.grabCut(orig_img, mask_segment, None, bgdm, fgdm, 10,
                                           cv2.GC_INIT_WITH_MASK)

    mask_segment = np.where((mask_segment == cv2.GC_BGD) | (mask_segment == cv2.GC_PR_BGD), 0,
                            1).astype(np.uint8)
    result1 = color_mask(mask_segment, color1)

    # Get full mask substract mask of segment1 to get segment2 mask
    mask_segfg2 = mask - mask_segment
    result2 = color_mask(mask_segfg2, color2)

    return cv2.add(result1, result2)


def run_grab_cut_by_mask1():
    # numpy.zeros -> Return a new array of given shape and type, filled with zeros.
    # create new mask with default value of probable foreground
    initial_mask = np.full(orig_img.shape[:2], cv2.GC_PR_BGD, np.uint8)

    # set sure foreground and background segments
    segfg = np.concatenate((seg0, seg1), axis=0)
    segbg = np.concatenate((seg2, seg3), axis=0)

    # set sure foreground
    for col, row in segfg:
        initial_mask[row][col] = cv2.GC_FGD

    # set sure background
    for col, row in segbg:
        initial_mask[row][col] = cv2.GC_BGD

    # bdgModel, fgdModel - These are arrays used by the grabCut algorithm
    # internally.two np.float64 type zero arrays of size (1,65).
    bgdm = np.zeros((1, 65), np.float64)
    fgdm = np.zeros((1, 65), np.float64)
    mask_fg, res_bgd, res_fgd = cv2.grabCut(orig_img, initial_mask, None, bgdm, fgdm, 10, cv2.GC_INIT_WITH_MASK)

    # numpy.where -> Return elements chosen from x or y depending on condition.
    mask_fg = np.where((mask_fg == cv2.GC_BGD) | (mask_fg == cv2.GC_PR_BGD), 0,
                       1).astype(np.uint8)

    result_mask_img = subtract_grab_cut(seg0, mask_fg, red_color, green_color)

    mask_bg = np.where((mask_fg == cv2.GC_PR_FGD) | (mask_fg == cv2.GC_FGD), cv2.GC_BGD,
                       cv2.GC_FGD).astype(np.uint8)

    result_mask_img = result_mask_img + subtract_grab_cut(seg2, mask_bg, blue_color, yellow_color)

    return result_mask_img


def main():
    global orig_img, seg_img, current_segment
    global seg0, seg1, seg2, seg3
    orig_img = cv2.imread(inputImage)
    seg_img = cv2.imread(inputImage)
    cv2.namedWindow("Select segments")

    # mouse event listener
    cv2.setMouseCallback("Select segments", mouse_click)
    # lists to hold pixels in each segment
    seg0 = []
    seg1 = []
    seg2 = []
    seg3 = []
    # segment you're on
    current_segment = 0
    while True:
        cv2.imshow("Select segments", seg_img)
        k = cv2.waitKey(20)

        # space bar to switch between segments
        if k == 32:
            current_segment = (current_segment + 1) % 4
            print('current segment = ' + str(current_segment))
        # escape
        if k == 27:
            break

    if len(seg0) == 0 or len(seg1) == 0 or len(seg2) == 0 or len(seg3) == 0:
        print("One or more segments are empty, Please add points to all segments")
        return

    segmaskImage = run_grab_cut_by_mask1()


    # creating the transparent image
    segmentedImage = cv2.addWeighted(orig_img, 0.6, segmaskImage, 0.4, 0)

    cv2.imwrite('segmentedImage.jpg', segmentedImage)
    cv2.imwrite('segments_mask_image.jpg', segmaskImage)
    cv2.imwrite('interfaceImg.jpg', seg_img)

    cv2.imshow("orig_image", orig_img)
    cv2.imshow("colored_imaged", segmaskImage)
    cv2.imshow("segmentedImage", segmentedImage)
    cv2.waitKey(0)

    # destroy all windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
