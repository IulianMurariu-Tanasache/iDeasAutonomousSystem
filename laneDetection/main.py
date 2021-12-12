import math
import cv2 as cv
import numpy as np

cap = cv.VideoCapture('drive.mp4')

# Check if camera opened successfully
if not cap.isOpened():
    print("Error opening video  file")

# Read until video is completed
while cap.isOpened():

    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:

        # gray -> gauss -> thresh -> canny
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)
        otsu_thresh, thresh = cv.threshold(blurred, 0, 255,
                                           cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        edges = cv.Canny(blurred, otsu_thresh * 0.5, otsu_thresh)

        # calculate vertices for trapezoid that masks the road
        rows, cols = edges.shape[:2]
        height_of_mask = rows * 0.69 # funny number nice
        bottom_left = [cols * 0.05, rows * 0.95]
        top_left = [cols * 0.45, height_of_mask]
        bottom_right = [cols * 0.85, rows]
        top_right = [cols * 0.55, height_of_mask]
        vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

        # mask the edges frame with the trapezoid
        mask = np.zeros_like(edges)
        ignore_mask_color = 255
        cv.fillPoly(mask, vertices, ignore_mask_color)
        masked_img = cv.bitwise_and(edges, mask)

        # get lines
        lines = cv.HoughLines(masked_img, 1, np.pi / 180, 150, None, 0, 0)

        # needed to filter multiple lines on one side of the road
        is_left = False
        is_right = False
        line_left = ((-9999999, 0), (-999999, 0))
        line_right = ((9999999, 0), (999999, 0))

        if lines is not None:
            for i in range(0, len(lines)):
                # compute the lines
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))

                # choose the line closest to the middle of the road
                pt = pt1 if pt1[1] > pt2[1] else pt2
                if pt[0] <= cols * 0.5:
                    is_left = True
                    if pt[0] > line_left[1][0]:
                        line_left = (pt1, pt2)
                else:
                    is_right = True
                    if pt[0] < line_right[0][0]:
                        line_right = (pt2, pt1)
        if is_left:
            # crop the line to the height of the mask
            line_left = ((int(((line_left[0][0] - line_left[1][0]) * (height_of_mask - line_left[0][1])) / (
                    line_left[0][1] - line_left[1][1]) + line_left[0][0]), int(height_of_mask)), line_left[0])
            cv.line(frame, line_left[0], line_left[1], (0, 0, 255), 3, cv.LINE_AA)
        if is_right:
            # crop the line to the height of the mask
            line_right = ((int(((line_right[0][0] - line_right[1][0]) * (height_of_mask - line_right[0][1])) / (
                    line_right[0][1] - line_right[1][1]) + line_right[0][0]), int(height_of_mask)), line_right[0])
            cv.line(frame, line_right[0], line_right[1], (0, 0, 255), 3, cv.LINE_AA)

        # Display the resulting frame
        cv.imshow('Frame', frame)

        # Press Q on keyboard to exit
        if cv.waitKey(25) & 0xFF == ord('q'):
            break
        # close window
        if cv.getWindowProperty('Frame', cv.WND_PROP_VISIBLE) < 1:
            break

    # Break the loop
    else:
        break

# When everything done, release
# the video capture object
cap.release()

# Closes all the frames
cv.destroyAllWindows()
