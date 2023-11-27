import cv2
import time
import numpy as np

# Capture from built-in video cam
cap = cv2.VideoCapture(0)

# Capture from IP server
# ip_camera_url = 'http://192.168.1.72:8080/video?type=some.mjpeg'
# cap = cv2.VideoCapture(ip_camera_url)

time_last = time.time()
time_last_print = time_last
frame_count = 0
frame_count_print = 30

while True:
    frame_count += 1
    ret, frame = cap.read()
    if not ret:
        print("Error reading from the camera.")
        break
    
    # Mark the brightest spot
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Built-in
    _, _, _, max_loc = cv2.minMaxLoc(gray)
    cv2.circle(frame, max_loc, 10, (0, 255, 255), 2)

    # Loop over pixels
    # rows, cols = gray.shape
    # max_intensity = 0
    # max_loc = (0, 0)
    # for row in range(rows):
    #     for col in range(cols):
    #         intensity = gray[row, col]
    #         if intensity > max_intensity:
    #             max_intensity = intensity
    #             max_loc = (col, row)
    # cv2.circle(frame, max_loc, 10, (0, 255, 255), 2)
    
    # Mark the reddest spot
    lower_red = np.array([0, 0, 100])
    upper_red = np.array([100, 100, 255])
    mask = cv2.inRange(frame, lower_red, upper_red)
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    masked_gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
    _, _, _, masked_max_loc = cv2.minMaxLoc(masked_gray)
    cv2.circle(frame, masked_max_loc, 10, (0, 0, 255), 2)
                
    # Calculate FPS
    time_now = time.time()
    fps = 1 / (time_now - time_last)
    if frame_count % frame_count_print == 0:
        print(f'Seconds per frame: {(time_now - time_last_print) / frame_count_print}')
        time_last_print = time_now
    cv2.putText(frame, f'FPS: {round(fps, 2)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    time_last = time_now

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
