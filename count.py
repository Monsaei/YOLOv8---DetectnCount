import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import*
import cvzone

model=YOLO('weights\yolov8n.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap=cv2.VideoCapture(r'D:\Documents\Cos\Fourth Year\Second Semester\Final Thesis\DetectTrackCount\videos\traffic1.mp4')

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


if not cap.isOpened():
        print("cannot open camera")
        exit()

count=0
cy1=424


tracker1=Tracker()
tracker2=Tracker()
tracker3=Tracker()



counter1=[]
counter2=[]
counter3=[]
offset=6
while True:
    ret,frame = cap.read()
    if not ret:
        break

    frame=cv2.resize(frame,(1020,500))


    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
#    print(px)
    list1=[]
    vehicle = []
    for index,row in px.iterrows():
#        print(row)

        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=classNames[d]
        if 'car'or'bus'or'truck'or'train' in c:
            list1.append([x1,y1,x2,y2])
            vehicle.append(c)

    bbox1_idx=tracker1.update(list1)

    for bbox1 in bbox1_idx:
        for i in vehicle:
            x3,y3,x4,y4,id1=bbox1
            cxm=int(x3+x4)//2
            cym=int(y3+y4)//2
            cv2.circle(frame,(cxm,cym),4,(0,0,255),-1)
            cv2.putText(frame,'Vehicle',(cxm,cym),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
            if cym<(cy1+offset) and cym>(cy1-offset):
               cv2.circle(frame,(cxm,cym),4,(0,255,0),-1)
               cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),1)
               cvzone.putTextRect(frame,f'{id1}',(x3,y3),1,1)
               if counter1.count(id1)==0:
                  counter1.append(id1)
   

    cv2.line(frame,(2,cy1),(794,cy1),(0,0,255),2)

  
    vehicle=(len(counter1))
    cvzone.putTextRect(frame,f'vehicle:-{vehicle}',(19,30),2,1)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(0)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()

