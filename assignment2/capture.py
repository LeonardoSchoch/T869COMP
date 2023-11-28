import cv2
import time
import numpy as np

# Hyperparameters ----------
cap = cv2.VideoCapture(0)
time_last = time.time()
frame_count = 0

frame_scale_factor = 1.0
canny_lower_threshold = 50
canny_upper_threshold = 150
ransac_iters = 250
ransac_delta = 1.0
k = 10
# --------------------------

while True:
    frame_count += 1
    ret, frame = cap.read()
    if not ret:
        print("Error reading from the camera.")
        break
    
    resized_frame = cv2.resize(frame, None, fx=frame_scale_factor, fy=frame_scale_factor)
    edges = cv2.Canny(resized_frame, canny_lower_threshold, canny_upper_threshold)
    edge_points_yx = np.column_stack(np.where(edges > 0))
    edge_points_xy = edge_points_yx[:, [1, 0]]
    edge_points_xy = edge_points_xy[::k]
    
    # Apply RANSAC to fit a line
    best_line = None
    max_inliers = 0
    if edge_points_xy.shape[0] >= 2:
        for _ in range(ransac_iters):
            sample_points = edge_points_xy[np.random.choice(len(edge_points_xy), 2, replace=False)]
            sample_line = cv2.fitLine(sample_points, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy, x, y = sample_line
            # point to line distance = abs(ax0 + by0 + c) / sqrt(a^2 + b^2)
            sample_distances = np.abs(vx * (edge_points_xy[:, 1] - y) - vy * (edge_points_xy[:, 0] - x)) \
                / np.sqrt(vx**2 + vy**2)
            sample_inliers = np.sum(sample_distances < ransac_delta)

            if sample_inliers > max_inliers:
                max_inliers = sample_inliers
                best_line = sample_line
            
    if best_line is not None:
        x0 = int(best_line[2] - best_line[0] * 1000)
        y0 = int(best_line[3] - best_line[1] * 1000)
        x1 = int(best_line[2] + best_line[0] * 1000)
        y1 = int(best_line[3] + best_line[1] * 1000)
        cv2.line(frame, (x0, y0), (x1, y1), (0, 255, 255), 2)
        
    # Calculate FPS
    time_now = time.time()
    fps = 1 / (time_now - time_last)
    cv2.putText(frame, f'FPS: {round(fps, 2)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    time_last = time_now

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
