import cv2

# Load pre-trained DNN model
model = cv2.dnn.readNetFromDarknet(
    "yolov3.cfg", "yolov3.weights"
)

# Load class labels
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Load traffic image
img = cv2.imread("traffic.jpg")
height, width = img.shape[:2]

# Preprocess image
blob = cv2.dnn.blobFromImage(
    img, 1/255.0, (416, 416), swapRB=True, crop=False
)

model.setInput(blob)

# Detect objects
layers = model.getUnconnectedOutLayersNames()
outputs = model.forward(layers)

# Draw bounding boxes
for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = scores.argmax()
        confidence = scores[class_id]

        if confidence > 0.5:
            box = detection[:4] * [width, height, width, height]
            x, y, w, h = box.astype("int")

            if classes[class_id] in [
                "car", "bus", "truck", "motorbike", "person"
            ]:
                cv2.rectangle(
                    img, (x-w//2, y-h//2),
                    (x+w//2, y+h//2), (0, 255, 0), 2
                )

                label = classes[class_id]
                cv2.putText(
                    img, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2
                )

# Display result
cv2.imshow("Traffic Object Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()