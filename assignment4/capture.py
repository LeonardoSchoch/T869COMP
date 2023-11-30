import cv2
import time
import numpy as np

# Hyperparameters ----------
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('https://192.168.1.72:8080/video')
time_last = time.time()
paused = False

frame_scale_factor = 1.0
# --------------------------

# Constants.
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.45

# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

# Colors
BLACK  = (0,0,0)
BLUE   = (255,178,50)
YELLOW = (0,255,255)
RED = (0,0,255)


def draw_label(input_image, label, left, top):
    """Draw text onto image at location."""
    
    # Get text size.
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    # Use text size to create a BLACK rectangle. 
    cv2.rectangle(input_image, (left, top), (left + dim[0], top + dim[1] + baseline), BLACK, cv2.FILLED);
    # Display text inside the rectangle.
    cv2.putText(input_image, label, (left, top + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)


def pre_process(input_image, net):
	# Create a 4D blob from a frame.
	blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), [0,0,0], 1, crop=False)

	# Sets the input to the network.
	net.setInput(blob)

	# Runs the forward pass to get output of the output layers.
	output_layers = net.getUnconnectedOutLayersNames()
	outputs = net.forward(output_layers)
	# print(outputs[0].shape)

	return outputs


def post_process(input_image, outputs):
    class_ids = []
    confidences = []
    boxes = []

    rows = outputs[0].shape[1]

    image_height, image_width = input_image.shape[:2]

    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT

    for r in range(rows):
        row = outputs[0][0][r]
        confidence = row[4]

        if confidence >= CONFIDENCE_THRESHOLD:
            classes_scores = row[5:]
            class_id = np.argmax(classes_scores)

            if classes_scores[class_id] > SCORE_THRESHOLD:
                confidences.append(confidence)
                class_ids.append(class_id)

                cx, cy, w, h = row[:4]

                left, top, width, height = map(int, ((cx - w/2) * x_factor, (cy - h/2) * y_factor, w * x_factor, h * y_factor))
                boxes.append([left, top, width, height])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    for i in indices:
        left, top, width, height = boxes[i]
        cv2.rectangle(input_image, (left, top), (left + width, top + height), BLUE, 3*THICKNESS)
        label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])
        draw_label(input_image, label, left, top)

    return input_image



classesFile = "coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

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
    

    # Give the weight files to the model and load the network using them.
    modelWeights = "models/yolov5n.onnx"
    net = cv2.dnn.readNet(modelWeights)

    # Process image.
    detections = pre_process(frame, net)
    img = post_process(frame.copy(), detections)

    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    print(label)
    # cv2.putText(img, label, (20, 40), FONT_FACE, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)
    
    # calculate FPS and print it on frame
    time_now = time.time()
    fps = 1 / (time_now - time_last)
    cv2.putText(img, f'FPS: {round(fps, 2)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    time_last = time_now
    
    cv2.imshow('Output', img)

cap.release()
cv2.destroyAllWindows()
