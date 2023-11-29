import cv2
import time
import numpy as np

# Hyperparameters ----------
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('https://192.168.1.72:8080/video')
time_last = time.time()
paused = False

frame_scale_factor = 1.0
canny_lower_threshold = 250
canny_upper_threshold = 500
k = 1

hough_rho = 1
hough_theta = np.pi / 180
hough_threshold = 100
hough_min_line_length = 250
hough_max_line_gap = 500

target_width = 720
target_height = 480
# --------------------------

def points_2_line(points):
    x1 = np.append(points[:2], 1)
    x2 = np.append(points[2:], 1)
    return np.cross(x1, x2)

def find_corners(lines, shape_frame):
    intersects = []
    for i in range(len(lines)):
        for j in range(len(lines)):
            if i != j:
                l1 = points_2_line(lines[i])
                l2 = points_2_line(lines[j])
                x12 = np.cross(l1, l2)
                if x12[2] == 0:
                    continue
                else:
                    x12 = x12/x12[2]
                    if abs(x12[0]) < shape_frame[1] and abs(x12[1]) < shape_frame[0]:
                        if list(x12[:2]) not in intersects:
                            intersects.append(list(x12[:2]))
    return np.array([[int(j) for j in i] for i in intersects])

def order_corners(corners):
    rect = np.zeros((4, 2))
    s = corners.sum(axis=1)
    rect[0] = corners[np.argmin(s)]
    rect[2] = corners[np.argmax(s)]
    diff = np.diff(corners, axis=1)
    rect[1] = corners[np.argmin(diff)]
    rect[3] = corners[np.argmax(diff)]
    return rect



while True:
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        paused = not paused
    if paused:
        continue
    
    ret, frame = cap.read()
    if not ret:
        print("Error reading from the camera.")
        break
    
    # resize frame
    frame = cv2.resize(frame, None, fx=frame_scale_factor, fy=frame_scale_factor)
    
    # detect edge points with Canny
    edges = cv2.Canny(frame, canny_lower_threshold, canny_upper_threshold)
    edge_points_yx = np.column_stack(np.where(edges > 0))
    edge_points_xy = edge_points_yx[:, [1, 0]]
    edge_points_xy = edge_points_xy[::k]
    
    # detect 4 most prominent lines with Hough and print them on frame
    lines = cv2.HoughLinesP(edges, hough_rho, hough_theta, hough_threshold, None, hough_min_line_length, hough_max_line_gap).squeeze()
    
    if lines is not None:
        selected_lines = lines[:4]
        for line in selected_lines:
            x1, y1, x2, y2 = line
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        corners = find_corners(selected_lines, frame.shape)
        for corner in corners:
            cv2.circle(frame, corner, 5, (0, 0, 255), -1)
        
        try:
            ordered_corners = order_corners(corners)
            target_rectangle = np.array([[0, 0],
                                        [target_width, 0],
                                        [target_width, target_height],
                                        [0, target_height]])
            h, status = cv2.findHomography(ordered_corners, target_rectangle)
            rectified = cv2.warpPerspective(frame, h, (target_width, target_height))
            
            cv2.imshow('rectified', rectified)
        except:
            pass
    
    # calculate FPS and print it on frame
    time_now = time.time()
    fps = 1 / (time_now - time_last)
    cv2.putText(frame, f'FPS: {round(fps, 2)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    time_last = time_now
    
    cv2.imshow("edges", edges)
    cv2.imshow("frame", frame)

cap.release()
cv2.destroyAllWindows()
