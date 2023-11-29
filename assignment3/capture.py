import cv2
import time
import numpy as np

# Hyperparameters ----------
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('https://192.168.1.72:8080/video')

time_last = time.time()
frame_count = 0

frame_scale_factor = 1.0
canny_lower_threshold = 250
canny_upper_threshold = 500
k = 1

# Hough Transform parameters
hough_rho = 1
hough_theta = np.pi / 180
hough_threshold = 100
hough_min_line_length = 200
hough_max_line_gap = 800
# --------------------------

lines_count = 0
while True:
    frame_count += 1
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
    lines = cv2.HoughLinesP(edges, hough_rho, hough_theta, hough_threshold, np.array([]), hough_min_line_length, hough_max_line_gap).squeeze()
    
    if lines is not None:
        selected_lines = lines[:4]
        for line in selected_lines:
            x1, y1, x2, y2 = line
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
    # calculate FPS and print it on frame
    time_now = time.time()
    fps = 1 / (time_now - time_last)
    cv2.putText(frame, f'FPS: {round(fps, 2)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    time_last = time_now
    
    # resize = ResizeWithAspectRatio(frame, width=1000)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()