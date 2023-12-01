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
    rows = outputs[0].shape[1]

    image_height, image_width = input_image.shape[:2]

    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT

    boxes = outputs[0][0, :, :4]
    boxes[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * x_factor
    boxes[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * y_factor
    boxes[:, 2:4] *= np.array([x_factor, y_factor])

    confidences = outputs[0][0, :, 4]
    indices = (confidences >= CONFIDENCE_THRESHOLD).nonzero()[0]

    boxes = boxes[indices]
    confidences = confidences[indices]

    class_ids = np.argmax(outputs[0][0, indices, 5:], axis=1)
    class_mask = (outputs[0][0, indices, 5 + class_ids] > SCORE_THRESHOLD)
    indices = indices[class_mask]

    boxes = boxes[class_mask]
    confidences = confidences[class_mask]
    class_ids = class_ids[class_mask]

    indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    for i in indices:
        box = boxes[i].astype(int)
        left, top, width, height = box
        cv2.rectangle(input_image, (left, top), (left + width, top + height), BLUE, 3 * THICKNESS)
        label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])
        draw_label(input_image, label, left, top)

    return input_image



# Initialize the model and read classes file outside the loop
modelWeights = "models/yolov5m.onnx"
net = cv2.dnn.readNet(modelWeights)

classesFile = "coco.names"
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

while True:
    start_time = cv2.getTickCount()

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

    # Process image.
    detections = pre_process(frame, net)
    img = post_process(frame.copy(), detections)

    # Put efficiency information.
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    print(label)
    
    # Calculate FPS and print it on frame
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - start_time)
    cv2.putText(img, f'FPS: {round(fps, 2)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow('Output', img)

cap.release()
cv2.destroyAllWindows()
