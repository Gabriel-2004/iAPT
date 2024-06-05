import cv2
import numpy as np
import pyzed.sl as sl
import torch
import time
import threading

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Create a ZED camera object
zed = sl.Camera()

# Set configuration parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD1080 
init_params.camera_fps = 30 
init_params.coordinate_units = sl.UNIT.METER

# Open the camera
err = zed.open(init_params)

if err != sl.ERROR_CODE.SUCCESS:
    print(f"Error opening ZED camera: {err}")
    exit(1)

runtime_parameters = sl.RuntimeParameters()
mat = sl.Mat()
point_cloud = sl.Mat()

# Variables for FPS calculation
fps = 0
frame_count = 0
start_time = time.time()

# Threading for YOLOv5
frame_lock = threading.Lock()
yolo_results = []

def yolo_thread(frame):
    global yolo_results
    with frame_lock:
        results = model(frame)
        yolo_results = results.xyxy[0].cpu().numpy()

try:
    while True:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(mat, sl.VIEW.LEFT)
            frame = mat.get_data()

            # Start YOLOv5 detection thread
            threading.Thread(target=yolo_thread, args=(frame,)).start()

            # Display results
            with frame_lock:
                for *box, conf, cls in yolo_results:
                    x1, y1, x2, y2 = map(int, box)
                    label = model.names[int(cls)]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                    # Calculate distance
                    zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
                    err, point = point_cloud.get_value((x1 + x2) // 2, (y1 + y2) // 2)
                    if err == sl.ERROR_CODE.SUCCESS:
                        distance = np.linalg.norm(point[:3])
                        cv2.putText(frame, f'Distance: {distance:.2f} m', (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Calculate and display FPS
            frame_count += 1
            if time.time() - start_time >= 1:
                fps = frame_count
                frame_count = 0
                start_time = time.time()
            cv2.putText(frame, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('ZED YOLOv5', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    # Release resources
    zed.close()
    cv2.destroyAllWindows()
