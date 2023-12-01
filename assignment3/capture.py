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

target_width = 480
target_height = 720
# --------------------------

def remove_similar_lines(lines):
    indices_to_remove = set()
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            x11, y11, x12, y12 = lines[i]
            x21, y21, x22, y22 = lines[j]
            v1 = np.array([x12 - x11, y12 - y11])
            v2 = np.array([x22 - x21, y22 - y21])
            dot_product = np.dot(v1, v2)
            if abs(dot_product) > 0.95 * np.linalg.norm(lines[i]) * np.linalg.norm(lines[j]):
                indices_to_remove.add(i)
    return np.delete(lines, indices_to_remove)

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
                    x12 = x12 / x12[2]
                    if abs(x12[0]) < shape_frame[1] and abs(x12[1]) < shape_frame[0]:
                        if list(x12[:2]) not in intersects:
                            intersects.append(list(x12[:2]))
    return np.array([[int(j) for j in i] for i in intersects])

def order_corners(corners):
    order = np.zeros((4, 2))
    sum = np.sum(corners, axis=1)
    order[0] = corners[np.argmin(sum)]
    order[2] = corners[np.argmax(sum)]
    diff = np.diff(corners, axis=1)
    order[1] = corners[np.argmin(diff)]
    order[3] = corners[np.argmax(diff)]
    return order



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
    
    # detect most prominent lines with Hough
    lines = cv2.HoughLinesP(edges, hough_rho, hough_theta, hough_threshold, None, hough_min_line_length, hough_max_line_gap).squeeze()
    
    # try to draw lines
    try:
        # selected_lines = remove_similar_lines(lines)
        selected_lines = lines[:4]
        for line in selected_lines:
            x1, y1, x2, y2 = line 
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    except Exception as e:
        print(f"{e}")
       
    # try to draw corners
    try:
        corners = find_corners(selected_lines, frame.shape)
        for corner in corners:
            cv2.circle(frame, corner, 5, (255, 0, 0), -1)
    except Exception as e:
        print(f"{e}")
        
    # try to rectify frame
    try:
        ordered_corners = order_corners(corners)
        target_rectangle = np.array([[0, 0],
                                    [target_width, 0],
                                    [target_width, target_height],
                                    [0, target_height]])
        h, status = cv2.findHomography(ordered_corners, target_rectangle)
        rectified = cv2.warpPerspective(frame, h, (target_width, target_height))
        
        cv2.imshow('rectified', rectified)
    except Exception as e:
        print(f"{e}")
    
    # calculate FPS and print it on frame
    time_now = time.time()
    fps = 1 / (time_now - time_last)
    cv2.putText(frame, f'FPS: {round(fps, 2)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    time_last = time_now
    
    cv2.imshow("edges", edges)
    cv2.imshow("frame", frame)

cap.release()
cv2.destroyAllWindows()
