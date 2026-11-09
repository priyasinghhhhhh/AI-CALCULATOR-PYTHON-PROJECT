import ast
import math
import os
import queue
import re
import threading
import time
from datetime import datetime

import cv2
import mediapipe as mp
import pandas as pd


# -------------------- SETTINGS --------------------

CAMERA_INDEX = 0

HISTORY_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "calculation_history.csv"
)

PANEL_W, PANEL_H = 470, 650

BUTTONS = [
    ["sin", "cos", "tan", "sqrt", "log"],
    ["ln", "pi", "e", "x²", "!"],
    ["7", "8", "9", "/", "DEL"],
    ["4", "5", "6", "*", "AC"],
    ["1", "2", "3", "-", "("],
    ["0", ".", ")", "+", "="],
    ["HIST", "VOICE", "−", "×", "÷"],
]

BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a ** b,
    ast.Mod: lambda a, b: a % b,
}

UNARY = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}

FUNCS = {
    "sin": lambda x: math.sin(math.radians(x)),
    "cos": lambda x: math.cos(math.radians(x)),
    "tan": lambda x: math.tan(math.radians(x)),
    "sqrt": math.sqrt,
    "log": math.log10,
    "ln": math.log,
}

CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}


# -------------------- CALCULATIONS --------------------

def safe_evaluate(expression):

    expression = (
        expression.replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace("^", "**")
    )

    if not expression.strip():
        raise ValueError("Enter a calculation first.")

    tree = ast.parse(expression, mode="eval")

    def visit(node):

        if isinstance(node, ast.Expression):
            return visit(node.body)

        if isinstance(node, ast.Constant) and isinstance(
            node.value, (int, float)
        ):
            return node.value

        if isinstance(node, ast.Name) and node.id in CONSTANTS:
            return CONSTANTS[node.id]

        if isinstance(node, ast.BinOp) and type(node.op) in BINOPS:

            left = visit(node.left)
            right = visit(node.right)

            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Exponent is too large.")

            value = BINOPS[type(node.op)](left, right)

            if isinstance(value, complex) or not math.isfinite(float(value)):
                raise ValueError("Invalid result.")

            return value

        if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY:
            return UNARY[type(node.op)](visit(node.operand))

        if isinstance(node, ast.Call):

            if not isinstance(node.func, ast.Name) or node.keywords:
                raise ValueError("Unsupported expression.")

            if node.func.id == "fact" and len(node.args) == 1:

                number = visit(node.args[0])

                if (
                    not float(number).is_integer()
                    or number < 0
                    or number > 170
                ):
                    raise ValueError(
                        "Factorial requires an integer from 0 to 170."
                    )

                return math.factorial(int(number))

            if node.func.id in FUNCS and len(node.args) == 1:
                return FUNCS[node.func.id](visit(node.args[0]))

        raise ValueError("Unsupported expression.")

    return visit(tree)


def format_result(value):

    if isinstance(value, float) and abs(value - round(value)) < 1e-12:
        return str(int(round(value)))

    if isinstance(value, float):
        return f"{value:.10g}"

    return str(value)


def add_token(expression, token):

    if token == "DEL":
        return expression[:-1]

    if token == "AC":
        return ""

    if token == "x²":
        return expression + "**2"

    if token == "!":

        match = re.search(r"(\d+(?:\.\d+)?)$", expression)

        if match:
            number = match.group(1)

            return (
                expression[:match.start()]
                + f"fact({number})"
            )

        return expression

    if token in ("sin", "cos", "tan", "sqrt", "log", "ln"):
        return expression + token + "("

    if token in ("pi", "e"):
        return expression + token

    if token in ("HIST", "VOICE", "="):
        return expression

    token = {
        "×": "*",
        "÷": "/",
        "−": "-"
    }.get(token, token)

    return expression + token


# -------------------- HISTORY --------------------

def load_history():

    columns = ["timestamp", "expression", "result"]

    try:
        data = pd.read_csv(HISTORY_FILE)
        return data[columns]

    except (
        FileNotFoundError,
        ValueError,
        KeyError,
        pd.errors.EmptyDataError
    ):
        return pd.DataFrame(columns=columns)


def save_history(history, expression, result):

    new_row = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "expression": expression,
        "result": str(result)
    }])

    history = pd.concat(
        [history, new_row],
        ignore_index=True
    )

    history.to_csv(HISTORY_FILE, index=False)

    return history


# -------------------- VOICE OUTPUT --------------------

class Voice:

    def __init__(self):

        self.enabled = False
        self.module = None
        self.messages = queue.Queue()

        try:
            import pyttsx3

            self.module = pyttsx3

            threading.Thread(
                target=self._worker,
                daemon=True
            ).start()

        except ImportError:
            print(
                "Voice output unavailable. "
                "Install using: pip install pyttsx3"
            )

    def _worker(self):

        try:
            engine = self.module.init()

            while True:

                message = self.messages.get()

                if message is None:
                    break

                engine.say(message)
                engine.runAndWait()

        except Exception as error:

            print("Voice output error:", error)

            self.module = None
            self.enabled = False

    def toggle(self):

        if self.module is None:

            print(
                "Voice unavailable. "
                "Install using: pip install pyttsx3"
            )

            self.enabled = False
            return

        self.enabled = not self.enabled

        print(
            "Voice:",
            "ON" if self.enabled else "OFF"
        )

    def speak(self, message):

        if self.enabled and self.module is not None:
            self.messages.put(str(message))


# -------------------- CAMERA --------------------

def open_camera(camera_index):

    backends = [
        ("DirectShow", cv2.CAP_DSHOW),
        ("Media Foundation", cv2.CAP_MSMF),
        ("Automatic", cv2.CAP_ANY)
    ]

    for backend_name, backend in backends:

        print(
            f"Trying camera {camera_index} "
            f"with {backend_name}..."
        )

        camera = cv2.VideoCapture(
            camera_index,
            backend
        )

        if not camera.isOpened():

            camera.release()
            continue

        camera.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            1280
        )

        camera.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            720
        )

        for attempt in range(15):

            success, frame = camera.read()

            if success and frame is not None:

                print(
                    f"Camera ready using {backend_name}."
                )

                return camera, frame

            time.sleep(0.15)

        print(
            f"{backend_name} opened the camera "
            "but could not read a frame."
        )

        camera.release()

    raise RuntimeError(
        "Could not capture a webcam frame. "
        "Close Camera, Zoom or Teams, check Windows "
        "camera permissions, and try CAMERA_INDEX = 1."
    )


# -------------------- CALCULATOR DISPLAY --------------------

def make_layout(frame_width, frame_height):

    scale = min(
        1.0,
        (frame_width - 20) / PANEL_W,
        (frame_height - 20) / PANEL_H
    )

    scale = max(0.35, scale)

    panel_w = int(PANEL_W * scale)
    panel_h = int(PANEL_H * scale)

    button_w = int(80 * scale)
    button_h = int(58 * scale)

    gap = max(2, int(6 * scale))

    grid_x = int(16 * scale)
    grid_y = int(178 * scale)

    panel_x = max(
        5,
        frame_width - panel_w - 12
    )

    panel_y = max(
        5,
        (frame_height - panel_h) // 2
    )

    return {
        "scale": scale,
        "panel_x": panel_x,
        "panel_y": panel_y,
        "panel_w": panel_w,
        "panel_h": panel_h,
        "button_w": button_w,
        "button_h": button_h,
        "gap": gap,
        "grid_x": grid_x,
        "grid_y": grid_y
    }


def button_rectangle(layout, row, column):

    x = layout["panel_x"] + layout["grid_x"] + column * (
        layout["button_w"] + layout["gap"]
    )

    y = layout["panel_y"] + layout["grid_y"] + row * (
        layout["button_h"] + layout["gap"]
    )

    return (
        x,
        y,
        x + layout["button_w"],
        y + layout["button_h"]
    )


def find_button(point, layout):

    if point is None:
        return None

    for row_index, row in enumerate(BUTTONS):

        for column_index, label in enumerate(row):

            x1, y1, x2, y2 = button_rectangle(
                layout,
                row_index,
                column_index
            )

            if (
                x1 <= point[0] <= x2
                and y1 <= point[1] <= y2
            ):
                return label

    return None


def draw_calculator(
    frame,
    expression,
    result,
    cursor,
    history,
    show_history,
    voice_enabled
):

    height, width = frame.shape[:2]

    layout = make_layout(width, height)

    px = layout["panel_x"]
    py = layout["panel_y"]

    scale = layout["scale"]

    def text(
        message,
        position,
        size=0.5,
        color=(240, 240, 240),
        thickness=1
    ):

        cv2.putText(
            frame,
            str(message),
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            size * scale,
            color,
            max(1, int(thickness * scale)),
            cv2.LINE_AA
        )

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (px, py),
        (
            px + layout["panel_w"],
            py + layout["panel_h"]
        ),
        (24, 29, 42),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.88,
        frame,
        0.12,
        0,
        frame
    )

    cv2.rectangle(
        frame,
        (px, py),
        (
            px + layout["panel_w"],
            py + layout["panel_h"]
        ),
        (90, 160, 245),
        max(1, int(2 * scale))
    )

    text(
        "AI VIRTUAL CALCULATOR",
        (
            px + int(18 * scale),
            py + int(32 * scale)
        ),
        0.7,
        (120, 205, 255),
        2
    )

    text(
        "Point with index finger | Pinch to select",
        (
            px + int(18 * scale),
            py + int(57 * scale)
        ),
        0.38
    )

    cv2.rectangle(
        frame,
        (
            px + int(15 * scale),
            py + int(72 * scale)
        ),
        (
            px + layout["panel_w"] - int(15 * scale),
            py + int(158 * scale)
        ),
        (12, 16, 25),
        -1
    )

    text(
        expression[-30:] if expression else "0",
        (
            px + int(27 * scale),
            py + int(103 * scale)
        ),
        0.6
    )

    text(
        "= " + str(result)[-25:],
        (
            px + int(27 * scale),
            py + int(140 * scale)
        ),
        0.7,
        (100, 235, 170),
        2
    )

    hovered = find_button(cursor, layout)

    for row_index, row in enumerate(BUTTONS):

        for column_index, label in enumerate(row):

            x1, y1, x2, y2 = button_rectangle(
                layout,
                row_index,
                column_index
            )

            if label == "=":
                color = (65, 115, 85)

            elif label == "AC":
                color = (90, 65, 75)

            elif label in ("HIST", "VOICE"):
                color = (65, 75, 105)

            else:
                color = (54, 64, 82)

            if label == hovered:
                color = (80, 130, 190)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                -1
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (115, 135, 165),
                1
            )

            font_scale = (
                0.45 if len(label) > 3 else 0.6
            ) * scale

            font_scale = max(0.25, font_scale)

            label_size = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                1
            )[0]

            text_x = (
                x1
                + (layout["button_w"] - label_size[0]) // 2
            )

            text_y = (
                y1
                + (layout["button_h"] + label_size[1]) // 2
            )

            cv2.putText(
                frame,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (250, 250, 255),
                1,
                cv2.LINE_AA
            )

    text(
        f"Voice: {'ON' if voice_enabled else 'OFF'} | "
        f"History: {'ON' if show_history else 'OFF'}",
        (
            px + int(18 * scale),
            py + layout["panel_h"] - int(13 * scale)
        ),
        0.4
    )

    if show_history:

        box_w = min(
            int(350 * scale),
            width - 20
        )

        box_h = min(
            int(205 * scale),
            height - 20
        )

        cv2.rectangle(
            frame,
            (10, 10),
            (10 + box_w, 10 + box_h),
            (20, 25, 36),
            -1
        )

        text(
            "RECENT CALCULATIONS",
            (20, 35),
            0.48,
            (120, 205, 255)
        )

        recent = history.tail(5).iloc[::-1]

        for index, (_, row) in enumerate(recent.iterrows()):

            text(
                f"{str(row['expression'])[:18]} = "
                f"{str(row['result'])[:12]}",
                (
                    20,
                    65 + index * int(27 * scale)
                ),
                0.42
            )

    return layout


# -------------------- MAIN APPLICATION --------------------

def main():

    print("Starting AI Virtual Calculator...")

    camera = None
    hands = None

    window_name = "Professional AI Virtual Calculator"

    try:

        camera, first_frame = open_camera(CAMERA_INDEX)

        mp_hands = mp.solutions.hands

        hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.6
        )

        drawing = mp.solutions.drawing_utils

        history = load_history()
        voice = Voice()

        expression = ""
        result = "Ready"

        show_history = False
        was_pinching = False
        last_click = 0.0

        print("Calculator started.")

        print(
            "Point with index finger and "
            "pinch thumb + index to select."
        )

        print(
            "Keyboard shortcuts: "
            "H = history, V = voice, Q or ESC = quit."
        )

        next_frame = first_frame

        while True:

            if next_frame is not None:

                frame = next_frame
                next_frame = None
                ok = True

            else:

                ok, frame = camera.read()

            if not ok or frame is None:

                print(
                    "Warning: webcam frame could not "
                    "be read. Retrying..."
                )

                time.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)

            frame_height, frame_width = frame.shape[:2]

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            data = hands.process(rgb_frame)

            cursor = None
            pinching = False

            if data.multi_hand_landmarks:

                hand = data.multi_hand_landmarks[0]

                drawing.draw_landmarks(
                    frame,
                    hand,
                    mp_hands.HAND_CONNECTIONS
                )

                thumb_tip = hand.landmark[4]
                index_tip = hand.landmark[8]

                cursor = (
                    int(index_tip.x * frame_width),
                    int(index_tip.y * frame_height)
                )

                distance = math.hypot(
                    thumb_tip.x - index_tip.x,
                    thumb_tip.y - index_tip.y
                )

                pinching = distance < 0.045

                cv2.circle(
                    frame,
                    cursor,
                    max(
                        6,
                        int(12 * min(frame_width / 640, 1.5))
                    ),
                    (70, 240, 140) if pinching else (255, 180, 60),
                    3
                )

            layout = make_layout(
                frame_width,
                frame_height
            )

            selected = find_button(cursor, layout)

            current_time = time.monotonic()

            if (
                pinching
                and not was_pinching
                and current_time - last_click > 0.35
                and selected is not None
            ):

                if selected == "VOICE":

                    voice.toggle()

                elif selected == "HIST":

                    show_history = not show_history

                elif selected == "=":

                    try:

                        result = format_result(
                            safe_evaluate(expression)
                        )

                        history = save_history(
                            history,
                            expression,
                            result
                        )

                        voice.speak(
                            "The answer is " + result
                        )

                    except Exception as error:

                        result = "Error: " + str(error)

                else:

                    expression = add_token(
                        expression,
                        selected
                    )

                    result = "Ready"

                last_click = current_time

            was_pinching = pinching

            draw_calculator(
                frame,
                expression,
                result,
                cursor,
                history,
                show_history,
                voice.enabled
            )

            cv2.putText(
                frame,
                "Q/ESC: Quit  V: Voice  H: History",
                (15, frame_height - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            cv2.imshow(
                window_name,
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):

                break

            elif key == ord("v"):

                voice.toggle()

            elif key == ord("h"):

                show_history = not show_history

    finally:

        if camera is not None:
            camera.release()
if __name__ == "__main__":
    print("Starting AI Virtual Calculator...")
    main()