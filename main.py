from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2 
import numpy as np

from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)


model = YOLO("yolov8n.pt")

@app.get("/ping")
async def ping():
    return {"message": "pong"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results = model(frame)

    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]
        detections.append({
            "box": [x1, y1, x2, y2],
            "confidence": conf,
            "class": label
        })

    return {"detections": detections}

