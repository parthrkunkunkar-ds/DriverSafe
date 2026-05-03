import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
import os
import tensorflow as tf

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
EAR_THRESHOLD   = 0.25
CLOSED_FRAMES   = 48        # ~2 sec at 24fps
CNN_THRESHOLD   = 0.85       # below this → CNN says eye closed
TFLITE_MODEL    = "models/drivesafe_float16.tflite"
ALARM_SOUND     = "phase1/alarm.wav"
FONT            = cv2.FONT_HERSHEY_SIMPLEX

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# ─────────────────────────────────────────────
#  TFLITE MODEL SETUP
# ─────────────────────────────────────────────
def load_tflite_model(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"[DriveSafe] CNN model loaded: {path}")
    print(f"  Input shape : {input_details[0]['shape']}")
    print(f"  Output shape: {output_details[0]['shape']}")
    return interpreter, input_details, output_details

def cnn_predict_eye(interpreter, input_details, output_details, eye_img):
    """
    eye_img: BGR crop from OpenCV
    Returns: float 0.0-1.0 (1.0 = open, 0.0 = closed)
    """
    img = cv2.resize(eye_img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # (1, 64, 64, 3)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return float(output[0][0])  # 1.0 = open, 0.0 = closed

# ─────────────────────────────────────────────
#  EAR CALCULATION
# ─────────────────────────────────────────────
def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append(np.array([lm.x * w, lm.y * h]))
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C)

# ─────────────────────────────────────────────
#  EXTRACT EYE CROP FOR CNN
# ─────────────────────────────────────────────
def extract_eye_crop(frame, landmarks, eye_indices, w, h, padding=10):
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append((int(lm.x * w), int(lm.y * h)))

    x_coords = [p[0] for p in pts]
    y_coords = [p[1] for p in pts]

    x1 = max(0, min(x_coords) - padding)
    y1 = max(0, min(y_coords) - padding)
    x2 = min(w, max(x_coords) + padding)
    y2 = min(h, max(y_coords) + padding)

    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else None

# ─────────────────────────────────────────────
#  DRAW EYE CONTOUR
# ─────────────────────────────────────────────
def draw_eye(frame, landmarks, eye_indices, w, h, color):
    pts = np.array([
        (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
        for idx in eye_indices
    ], dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1)

# ─────────────────────────────────────────────
#  ALARM
# ─────────────────────────────────────────────
def init_alarm():
    pygame.mixer.init()
    if os.path.exists(ALARM_SOUND):
        return pygame.mixer.Sound(ALARM_SOUND)
    sample_rate = 44100
    duration    = 0.5
    t    = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    stereo = np.column_stack([wave, wave])
    return pygame.sndarray.make_sound(stereo)

def play_alarm(sound, is_playing):
    if not is_playing:
        sound.play(-1)
    return True

def stop_alarm(sound, is_playing):
    if is_playing:
        sound.stop()
    return False

# ─────────────────────────────────────────────
#  DRAW OVERLAY
# ─────────────────────────────────────────────
def draw_overlay(frame, ear, cnn_score, frame_count, alarm_active, detection_mode):
    h, w = frame.shape[:2]

    def put_text_bg(frame, text, pos, scale, color, thickness=2):
        """Draw text with dark background for visibility"""
        (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
        x, y = pos
        cv2.rectangle(frame, (x - 4, y - th - 4), (x + tw + 4, y + 6), (0, 0, 0), -1)
        cv2.putText(frame, text, (x, y), FONT, scale, color, thickness)

    # EAR value
    ear_color = (0, 255, 0) if ear > EAR_THRESHOLD else (0, 0, 255)
    put_text_bg(frame, f"EAR: {ear:.3f}", (30, 40), 0.7, ear_color)

    # CNN score
    cnn_color = (0, 255, 0) if cnn_score > CNN_THRESHOLD else (0, 0, 255)
    put_text_bg(frame, f"CNN: {cnn_score:.3f}", (30, 70), 0.7, cnn_color)

    # Detection mode
    put_text_bg(frame, f"Mode: {detection_mode}", (30, 100), 0.5, (255, 255, 255))

    # Drowsy meter label
    put_text_bg(frame, "Drowsy meter", (30, 115), 0.4, (200, 200, 200), 1)

    # Drowsy progress bar
    bar_w = min(int((frame_count / CLOSED_FRAMES) * 200), 200)
    cv2.rectangle(frame, (30, 120), (230, 135), (60, 60, 60), -1)
    cv2.rectangle(frame, (30, 120), (30 + bar_w, 135), (0, 0, 255), -1)

    # FPS
    fps_text = f"FPS: {1 / max(time.time() - time.time(), 1e-6):.1f}"

    if alarm_active:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h//2 - 60), (w, h//2 + 60), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.putText(frame, "DROWSY! PULL OVER!", (w//2 - 220, h//2 - 10),
                    FONT, 1.1, (255, 255, 255), 3)
        cv2.putText(frame, "Find a safe spot and rest", (w//2 - 175, h//2 + 30),
                    FONT, 0.7, (255, 255, 255), 2)

# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────
def main():
    # Load TFLite model
    interpreter, input_details, output_details = load_tflite_model(TFLITE_MODEL)

    # MediaPipe setup
    import mediapipe.python.solutions.face_mesh as mp_face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    # Camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    alarm_sound  = init_alarm()
    closed_count = 0
    alarm_active = False
    fps_time     = time.time()
    fps          = 0

    print("[DriveSafe] Phase 3 running — EAR + CNN ensemble active. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera read failed.")
            break

        h, w = frame.shape[:2]

        # FPS
        fps = 0.9 * fps + 0.1 * (1 / max(time.time() - fps_time, 1e-6))
        fps_time = time.time()
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 110, 30), FONT, 0.6, (180, 180, 180), 1)

        # MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = face_mesh.process(rgb)
        rgb.flags.writeable = True

        if results.multi_face_landmarks:
            lms = results.multi_face_landmarks[0].landmark

            # ── EAR ──
            left_ear  = eye_aspect_ratio(lms, LEFT_EYE,  w, h)
            right_ear = eye_aspect_ratio(lms, RIGHT_EYE, w, h)
            avg_ear   = (left_ear + right_ear) / 2.0
            ear_closed = avg_ear < EAR_THRESHOLD

            # ── CNN on left eye crop ──
            left_crop = extract_eye_crop(frame, lms, LEFT_EYE, w, h)
            if left_crop is not None:
                cnn_score = cnn_predict_eye(interpreter, input_details, output_details, left_crop)
            else:
                cnn_score = 1.0  # assume open if crop fails

            cnn_closed = cnn_score < CNN_THRESHOLD

            # ── Ensemble decision ──
            # Both must agree for alarm to fire
            both_closed = ear_closed or cnn_closed

            # Detection mode label for overlay
            if ear_closed and cnn_closed:
                detection_mode = "BOTH CLOSED"
                eye_color = (0, 0, 255)
            elif ear_closed:
                detection_mode = "EAR only"
                eye_color = (0, 165, 255)
            elif cnn_closed:
                detection_mode = "CNN only"
                eye_color = (0, 165, 255)
            else:
                detection_mode = "Eyes Open"
                eye_color = (0, 255, 0)

            draw_eye(frame, lms, LEFT_EYE,  w, h, eye_color)
            draw_eye(frame, lms, RIGHT_EYE, w, h, eye_color)

            if both_closed:
                closed_count += 1
                if closed_count >= CLOSED_FRAMES:
                    alarm_active = play_alarm(alarm_sound, alarm_active)
            else:
                closed_count = 0
                alarm_active = stop_alarm(alarm_sound, alarm_active)

            draw_overlay(frame, avg_ear, cnn_score, closed_count, alarm_active, detection_mode)

        else:
            cv2.putText(frame, "No face detected", (30, 40), FONT, 0.7, (0, 165, 255), 2)
            closed_count = 0
            alarm_active = stop_alarm(alarm_sound, alarm_active)

        cv2.imshow("DriveSafe — Phase 3 (EAR + CNN)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
    print("[DriveSafe] Session ended.")

if __name__ == "__main__":
    main()