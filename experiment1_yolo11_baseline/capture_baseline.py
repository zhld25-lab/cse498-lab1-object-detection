import csv, os, re, sys
from datetime import datetime
import cv2
from ultralytics import YOLO

MODEL_PATH = "./models/yolo11n.pt"
CAM_INDEX = 1
CONF = 0.25
OUT_DIR = "results/exp1_baseline"
RAW_DIR = os.path.join(OUT_DIR, "raw")
ANN_DIR = os.path.join(OUT_DIR, "annotated")
CSV_PATH = os.path.join(OUT_DIR, "detections.csv")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(ANN_DIR, exist_ok=True)
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["image_id","timestamp","ground_truth","predicted_class","confidence","x1","y1","x2","y2","conf_threshold","correct","notes"])

def next_index():
    used = []
    for name in os.listdir(RAW_DIR):
        m = re.fullmatch(r"test(\d+)\.jpg", name)
        if m:
            used.append(int(m.group(1)))
    return max(used) + 1 if used else 1

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    sys.exit("Could not open camera. Close Zoom/Teams and retry.")

shot = next_index()
print("Ready. Next capture: test%02d. Press s to save, q to quit." % shot)

while True:
    ok, frame = cap.read()
    if not ok:
        continue
    raw = frame.copy()
    result = model.predict(frame, conf=CONF, verbose=False)[0]
    rows = []
    for box, score, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        label = model.names[int(cls)]
        score = float(score)
        color = (0, 255, 0) if score >= 0.5 else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, "%s %.2f" % (label, score), (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        rows.append([label, round(score, 4), x1, y1, x2, y2])
    hud = "thr=%.2f  dets=%d  next=test%02d  [s]ave [q]uit" % (CONF, len(rows), shot)
    cv2.putText(frame, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("YOLO11 baseline - Experiment 1", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("["):
        CONF = max(0.05, round(CONF - 0.05, 2))
    elif key == ord("]"):
        CONF = min(0.95, round(CONF + 0.05, 2))
    elif key == ord("s"):
        image_id = "test%02d" % shot
        stamp = datetime.now().isoformat(timespec="seconds")
        cv2.imwrite(os.path.join(RAW_DIR, image_id + ".jpg"), raw)
        cv2.imwrite(os.path.join(ANN_DIR, image_id + ".jpg"), frame)
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if rows:
                for label, score, x1, y1, x2, y2 in rows:
                    w.writerow([image_id, stamp, "", label, score, x1, y1, x2, y2, CONF, "", ""])
            else:
                w.writerow([image_id, stamp, "", "NO_DETECTION", "", "", "", "", "", CONF, "", ""])
        print("saved %s (%d detections, thr=%.2f)" % (image_id, len(rows), CONF))
        shot += 1

cap.release()
cv2.destroyAllWindows()
print("Done. Results in " + OUT_DIR)
