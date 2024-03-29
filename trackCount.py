from ultralytics import YOLO
import numpy as np
import cv2
import random

# classes from pretrained model
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"
                  ]

# visuals kineme sa bounding boxes
lgbtq_detection = []
for i in range(len(classNames)):
        r = random.randint(0,255)
        g = random.randint(0,255)
        b = random.randint(0,255)
        lgbtq_detection.append((b,g,r))

model = YOLO('../weights/yolov8n.pt', 'v8')

# vid vals
xFrame = 640
yFrame = 480

# cap = cv2.VideoCapture(1) //cam
cap = cv2.VideoCapture("videos/4K_Road_traffic_video_for_object_detection_and_tracking_-_free_download_now.mp4")

if not cap.isOpened():
        print("cannot open camera")
        exit()

while True:
        # frame capture
        ret, frame = cap.read()

        # if clause to automatically exit applciation after video input ends
        if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

         # predict laman ng source
        detect_params = model.predict(source=[frame], conf=0.45, save = False)

        DP = detect_params[0].numpy()
        print(DP)

        if len(DP) != 0:
            for i in range(len(detect_params[0])):
                print(i)

                boxes = detect_params[0].boxes
                box = boxes[i]  # returns one box
                clsID = box.cls.numpy()[0]
                conf = box.conf.numpy()[0]
                bb = box.xyxy.numpy()[0]

                cv2.rectangle(
                    frame,
                    (int(bb[0]), int(bb[1])),
                    (int(bb[2]), int(bb[3])),
                    lgbtq_detection[int(clsID)],
                    3,
                )

                # Display class name and confidence
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(
                    frame,
                    classNames[int(clsID)] + " " + str(round(conf, 3)) + "%",
                    (int(bb[0]), int(bb[1]) - 10),
                    font,
                    1,
                    (255, 255, 255),
                    2,
                )
        cv2.imshow("ObjectDetection", frame)
        if cv2.waitKey(1) == ord("q"):
            break
cap.release()
cv2.destroyAllWindows()
