# Professional AI Virtual Calculator

A beginner-friendly webcam calculator. Point with your index finger and pinch your thumb and index finger together to click a button.

## Features
- Webcam video with a floating calculator panel
- MediaPipe hand landmark tracking
- Pinch gesture selection
- Arithmetic and scientific functions: +, -, *, /, sin, cos, tan, sqrt, log, ln, powers, factorial, pi and e
- Optional voice output (`pyttsx3`)
- Pandas calculation history saved to `calculation_history.csv`

## Folder structure
```text
professional_ai_virtual_calculator/
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Recommended setup
Use Python 3.10 or 3.11. MediaPipe wheels vary by OS and Python version.

### Windows
```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

### macOS / Linux
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

Allow camera access if prompted. If the webcam does not open, close other camera apps or change `CAMERA_INDEX = 0` in `main.py` to `1`.

## How to use
1. Keep one hand visible to the webcam.
2. Move the index fingertip over a calculator button.
3. Pinch thumb and index finger together to select it.
4. Release the pinch before the next selection.
5. Press `=` to calculate, `AC` to clear, `DEL` to delete.
6. `H` or the `HIST` button toggles history; `V` or `VOICE` toggles speech.
7. Press `Q` or `Esc` to quit.

Voice is optional:
```bash
pip install pyttsx3
```
On some Linux systems a speech engine may also be needed. The calculator works without voice.

## Tips
- If pinching is too sensitive, change `0.045` in `pinching=...` to `0.035`.
- If it is too difficult to trigger, try `0.055`.
- Trigonometric functions use degrees.
- `log` is base 10; `ln` is natural logarithm.
- The expression evaluator only allows supported math syntax; it does not use Python `eval()`.

## Learning overview
OpenCV captures camera frames; MediaPipe detects hand landmarks; the program maps the index fingertip to a button; a pinch triggers the button; Python evaluates the math; Pandas saves each successful calculation to CSV.
