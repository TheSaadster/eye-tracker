
# Eye Tracker with Blink-to-Click

This project uses your **webcam, MediaPipe, OpenCV, and PyAutoGUI** to control your mouse with your eyes. After a quick calibration step, the program maps your gaze direction to screen coordinates and lets you **click by blinking**.

---

## 🚀 Features

* Real-time **eye tracking** using MediaPipe Face Mesh.
* **Calibration step** for more accurate screen mapping.
* **Blink detection** (close your eyes for a short moment to click).
* **Smoothing filter** for less jittery cursor movement.
* Runs on a normal webcam — no special hardware required.

---

## 🛠 Requirements

* Python 3.8+
* OpenCV
* MediaPipe
* PyAutoGUI
* NumPy

Install dependencies:

```bash
pip install opencv-python mediapipe pyautogui numpy
```

---

## ▶️ Usage

1. Run the script:

   ```bash
   python main.py
   ```

2. **Calibration**:

   * The program will move your cursor to corners and center of the screen.
   * Look at the point and press **SPACE** to record each calibration.
   * Press **ESC** to cancel.

3. **Tracking mode**:

   * Move your eyes → cursor follows.
   * Blink → performs a click.
   * Press **ESC** to exit.

---

## ⚙️ Controls

* **SPACE** → confirm calibration point.
* **ESC** → exit at any time.
* **Blink** (held for ~5 frames) → left mouse click.

---

## ⚠️ Notes

* Good lighting helps accuracy.
* Calibration is critical — look directly at the point before pressing **SPACE**.
* PyAutoGUI’s failsafe (top-left corner panic escape) is disabled in this version. If you want it back, remove:

  ```python
  pyautogui.FAILSAFE = False
  ```

---


