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

# Load two different YOLO models
# model_primary = YOLO("yolov8n.pt")  
model_secondary = YOLO("best.pt")    

@app.get("/ping")
async def ping():
    return {"message": "pong"}


# @app.post("/detect")
# async def detect(file: UploadFile = File(...)):
#     """Object detection using primary model (YOLOv8n)."""
#     contents = await file.read()
#     npimg = np.frombuffer(contents, np.uint8)
#     frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

#     results = model_primary(frame)

#     detections = []
#     for box in results[0].boxes:
#         x1, y1, x2, y2 = box.xyxy[0].tolist()
#         conf = float(box.conf[0])
#         cls = int(box.cls[0])
#         label = model_primary.names[cls]
#         detections.append({
#             "box": [x1, y1, x2, y2],
#             "confidence": conf,
#             "class": label
#         })

#     return {"model": "detect", "detections": detections}


@app.post("/detect")
async def detect_alt(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results = model_secondary(frame)

    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model_secondary.names[cls]
        detections.append({
            "box": [x1, y1, x2, y2],
            "confidence": conf,
            "class": label
        })

    return {"model": "v1", "detections": detections}



