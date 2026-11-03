"""Collect labeled hand-landmark samples from your webcam.
Run: python collect_data.py
Press 1-8 to select a label, SPACE to save a sample, Q to quit.
"""
import csv, os, time
import cv2
import mediapipe as mp

LABELS = {
    ord('1'): "HELLO",
    ord('2'): "THANK YOU",
    ord('3'): "YES",
    ord('4'): "NO",
    ord('5'): "HELP",
    ord('6'): "WATER",
    ord('7'): "PLEASE",
    ord('8'): "SORRY",
}
OUT = "sign_landmarks.csv"
mp_hands = mp.solutions.hands
drawer = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open webcam. Check Windows camera permissions.")

current_label = "HELLO"
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.55, min_tracking_confidence=0.5) as hands, \
     open(OUT, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    if os.path.getsize(OUT) == 0:
        writer.writerow(["label"] + [f"v{i}" for i in range(63)])
    print("Keys: 1 HELLO, 2 THANK YOU, 3 YES, 4 NO, 5 HELP, 6 WATER, 7 PLEASE, 8 SORRY")
    print("SPACE saves one sample for the selected label. Q quits.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Could not read webcam frame.")
            break
        frame = cv2.flip(frame, 1)
        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        landmarks = None
        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            drawer.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            landmarks = []
            for p in hand.landmark:
                landmarks.extend([p.x, p.y, p.z])
        cv2.rectangle(frame, (0,0), (frame.shape[1],80), (25,35,60), -1)
        cv2.putText(frame, f"LABEL: {current_label}", (12,30), cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 2)
        cv2.putText(frame, "1-8 choose label | SPACE save | Q quit", (12,62), cv2.FONT_HERSHEY_SIMPLEX, .55, (200,220,255), 1)
        cv2.imshow("SignSpeak Data Collector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in LABELS:
            current_label = LABELS[key]
        elif key == ord(' ') and landmarks is not None:
            writer.writerow([current_label] + landmarks)
            f.flush()
            print("Saved sample:", current_label)
            time.sleep(0.12)
        elif key == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
