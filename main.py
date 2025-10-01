import cv2
import mediapipe as mp
import numpy as np
import pyautogui

pyautogui.FAILSAFE = False

SMOOTHING_ALPHA = 0.2

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
cap = cv2.VideoCapture(0)

screen_w, screen_h = pyautogui.size()

blink_counter = 0
BLINK_FRAMES = 2

calibration_points = {}
gaze_transform = None
filtered_cursor = None


def get_iris(frame, landmarks):
    h, w, _ = frame.shape
    left_iris = landmarks[468]
    right_iris = landmarks[473]
    left_iris_x, left_iris_y = int(left_iris.x * w), int(left_iris.y * h)
    right_iris_x, right_iris_y = int(right_iris.x * w), int(right_iris.y * h)
    return (left_iris_x + right_iris_x) // 2, (left_iris_y + right_iris_y) // 2


def compute_gaze_transform():
    if len(calibration_points) < 3:
        return None

    src = []
    dst_x = []
    dst_y = []
    for data in calibration_points.values():
        iris_x, iris_y, screen_x, screen_y = data
        src.append([iris_x, iris_y, 1.0])
        dst_x.append(screen_x)
        dst_y.append(screen_y)

    src = np.array(src, dtype=np.float32)
    dst_x = np.array(dst_x, dtype=np.float32)
    dst_y = np.array(dst_y, dtype=np.float32)

    coeffs_x, _, _, _ = np.linalg.lstsq(src, dst_x, rcond=None)
    coeffs_y, _, _, _ = np.linalg.lstsq(src, dst_y, rcond=None)

    return np.vstack([coeffs_x, coeffs_y])


def calibrate():
    calibration_points.clear()

    positions = {
        "top-left": (100, 100),
        "top-right": (screen_w - 100, 100),
        "bottom-left": (100, screen_h - 100),
        "bottom-right": (screen_w - 100, screen_h - 100),
        "center": (screen_w // 2, screen_h // 2),
    }

    for label, pos in positions.items():
        pyautogui.moveTo(*pos)
        print(f"Look at the {label} corner and press SPACE...")
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            cv2.putText(
                frame,
                f"Look at {label} and press SPACE",
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Calibration", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 32 and results.multi_face_landmarks:
                iris_x, iris_y = get_iris(
                    frame, results.multi_face_landmarks[0].landmark
                )
                calibration_points[label] = (iris_x, iris_y, pos[0], pos[1])
                break
            elif key == 27:
                return False
    return True


def map_to_screen(x, y):
    if gaze_transform is None:
        return screen_w // 2, screen_h // 2

    vector = np.array([x, y, 1.0], dtype=np.float32)
    mapped = gaze_transform @ vector
    screen_x = int(np.clip(mapped[0], 0, screen_w - 1))
    screen_y = int(np.clip(mapped[1], 0, screen_h - 1))
    return screen_x, screen_y


print("Starting calibration...")
if not calibrate():
    cap.release()
    cv2.destroyAllWindows()
    raise SystemExit

gaze_transform = compute_gaze_transform()
if gaze_transform is None:
    print("Calibration failed. Not enough samples collected.")
    cap.release()
    cv2.destroyAllWindows()
    raise SystemExit

print("Calibration complete! Now tracking...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        iris_x, iris_y = get_iris(frame, landmarks)

        target_x, target_y = map_to_screen(iris_x, iris_y)

        if filtered_cursor is None:
            filtered_cursor = (target_x, target_y)
        else:
            filtered_cursor = (
                int(
                    filtered_cursor[0]
                    + SMOOTHING_ALPHA * (target_x - filtered_cursor[0])
                ),
                int(
                    filtered_cursor[1]
                    + SMOOTHING_ALPHA * (target_y - filtered_cursor[1])
                ),
            )

        pyautogui.moveTo(*filtered_cursor)

        left_eye_top = landmarks[159]
        left_eye_bottom = landmarks[145]
        eye_open_dist = abs(left_eye_top.y - left_eye_bottom.y)

        if eye_open_dist < 0.01:
            blink_counter += 1
        else:
            if blink_counter > BLINK_FRAMES:
                pyautogui.click()
            blink_counter = 0

        if blink_counter > 1:
            cv2.putText(
                frame,
                "Blinking...",
                (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        cv2.circle(frame, (iris_x, iris_y), 4, (0, 255, 0), -1)
        cv2.putText(
            frame,
            f"Cursor: ({target_x}, {target_y})",
            (30, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    cv2.imshow("Eye Tracker", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
