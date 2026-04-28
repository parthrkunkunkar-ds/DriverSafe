import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
import os

# ─────────────────────────────────────────────
#  CONFIG  — tweak these without touching logic
# ─────────────────────────────────────────────
EAR_THRESHOLD   = 0.25   # below this → eye considered closed
CLOSED_FRAMES   = 48     # ~2 sec at 24 fps before alarm fires
ALARM_SOUND     = "alarm.wav"  # place in same directory; fallback to beep
FONT            = cv2.FONT_HERSHEY_SIMPLEX

# MediaPipe landmark indices for LEFT and RIGHT eye
# Each list: [left_corner, top_outer, top_inner, right_corner, bot_inner, bot_outer]
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# ─────────────────────────────────────────────
#  EAR CALCULATION
# ─────────────────────────────────────────────
def eye_aspect_ratio(landmarks, eye_indices, w, h):
    """
    EAR = (‖p2−p6‖ + ‖p3−p5‖) / (2 × ‖p1−p4‖)
    Ranges ~0.30 (open) → ~0.0 (fully closed).
    """
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append(np.array([lm.x * w, lm.y * h]))

    # Vertical distances
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    # Horizontal distance
    C = np.linalg.norm(pts[0] - pts[3])

    return (A + B) / (2.0 * C)

# ─────────────────────────────────────────────
#  DRAW EYE CONTOUR  (visual debug aid)
# ─────────────────────────────────────────────
def draw_eye(frame, landmarks, eye_indices, w, h, color):
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append((int(lm.x * w), int(lm.y * h)))
    pts = np.array(pts, dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1)

# ─────────────────────────────────────────────
#  ALARM SETUP
# ─────────────────────────────────────────────
def init_alarm():
    pygame.mixer.init()
    if os.path.exists(ALARM_SOUND):
        return pygame.mixer.Sound(ALARM_SOUND)
    # Synthesise a 440Hz beep if no file found
    sample_rate = 44100
    duration    = 0.5
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    stereo = np.column_stack([wave, wave])
    sound = pygame.sndarray.make_sound(stereo)
    return sound

def play_alarm(sound, is_playing):
    if not is_playing:
        sound.play(-1)   # -1 = loop indefinitely
    return True

def stop_alarm(sound, is_playing):
    if is_playing:
        sound.stop()
    return False

# ─────────────────────────────────────────────
#  DRAW OVERLAY
# ─────────────────────────────────────────────
def draw_overlay(frame, ear, frame_count, alarm_active):
    h, w = frame.shape[:2]

    # EAR value
    ear_color = (0, 255, 0) if ear > EAR_THRESHOLD else (0, 0, 255)
    cv2.putText(frame, f"EAR: {ear:.3f}", (30, 40),
                FONT, 0.7, ear_color, 2)

    # Drowsy progress bar
    bar_w = int((frame_count / CLOSED_FRAMES) * 200)
    bar_w = min(bar_w, 200)
    cv2.rectangle(frame, (30, 55), (230, 70), (60, 60, 60), -1)
    cv2.rectangle(frame, (30, 55), (30 + bar_w, 70), (0, 0, 255), -1)
    cv2.putText(frame, "Drowsy meter", (30, 50),
                FONT, 0.4, (200, 200, 200), 1)

    if alarm_active:
        # Red flashing banner
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
    # MediaPipe setup
    mp_face  = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,   # enables iris + more precise eyelid
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    # Camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    alarm_sound  = init_alarm()
    closed_count = 0
    alarm_active = False
    fps_time     = time.time()
    fps          = 0

    print("[DriveSafe] Phase 1 running. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera read failed.")
            break

        h, w = frame.shape[:2]
        # FPS counter
        fps = 0.9 * fps + 0.1 * (1 / max(time.time() - fps_time, 1e-6))
        fps_time = time.time()
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 110, 30),
                    FONT, 0.6, (180, 180, 180), 1)

        # MediaPipe inference (RGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = face_mesh.process(rgb)
        rgb.flags.writeable = True

        if results.multi_face_landmarks:
            lms = results.multi_face_landmarks[0].landmark

            left_ear  = eye_aspect_ratio(lms, LEFT_EYE,  w, h)
            right_ear = eye_aspect_ratio(lms, RIGHT_EYE, w, h)
            avg_ear   = (left_ear + right_ear) / 2.0

            # Draw eye contours
            eye_color = (0, 255, 0) if avg_ear > EAR_THRESHOLD else (0, 0, 255)
            draw_eye(frame, lms, LEFT_EYE,  w, h, eye_color)
            draw_eye(frame, lms, RIGHT_EYE, w, h, eye_color)

            if avg_ear < EAR_THRESHOLD:
                closed_count += 1
                if closed_count >= CLOSED_FRAMES:
                    alarm_active = play_alarm(alarm_sound, alarm_active)
            else:
                closed_count = 0
                alarm_active = stop_alarm(alarm_sound, alarm_active)

            draw_overlay(frame, avg_ear, closed_count, alarm_active)

        else:
            cv2.putText(frame, "No face detected", (30, 40),
                        FONT, 0.7, (0, 165, 255), 2)
            closed_count = 0
            alarm_active = stop_alarm(alarm_sound, alarm_active)

        cv2.imshow("DriveSafe — Phase 1", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
    print("[DriveSafe] Session ended.")

if __name__ == "__main__":
    main()