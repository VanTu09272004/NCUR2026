import cv2
import mediapipe as mp
import math
import os
import numpy as np
import feedparser
from screeninfo import get_monitors
from dotenv import load_dotenv
import requests
from datetime import datetime, timedelta

#CONFIG / THEME
load_dotenv()
WEATHER_KEY = os.getenv("WEATHER_API_KEY")

FONT = cv2.FONT_HERSHEY_SIMPLEX

UI = {
    "bg": (10, 10, 10),
    "card_fill": (25, 25, 25),
    "border": (180, 180, 180),
    "text": (230, 230, 230),
    "muted": (160, 160, 160),
    "accent": (0, 220, 225),  #highlight
}

PAD_X = 30
TITLE_Y_OFFSET = 28        # where title baseline sits
CONTENT_TOP_OFFSET = 44    # y + this is where content begins
BOTTOM_PAD = 12

# Cursor styling
CURSOR_IDLE_COLOR = (128, 128, 128)
CURSOR_GRAB_COLOR = UI["accent"]
CURSOR_IDLE_R = 10
CURSOR_GRAB_R = 14


#WEATHER (cached)

last_loc = None
last_loc_time = None
last_weather = None
last_weather_time = None


def get_lat_lon():
    #Get approximate lat/lon by public IP; cache for 1 hour
    global last_loc, last_loc_time
    now = datetime.now()

    if last_loc and last_loc_time and now - last_loc_time < timedelta(hours=1):
        return last_loc

    resp = requests.get("http://ip-api.com/json/", timeout=5)
    data = resp.json()
    lat, lon = data["lat"], data["lon"]

    last_loc = (lat, lon)
    last_loc_time = now
    return last_loc


def get_weather():
    #Get weather from WeatherAPI; cache for 10 minutes
    global last_weather, last_weather_time
    now = datetime.now()

    if last_weather and last_weather_time and now - last_weather_time < timedelta(minutes=10):
        return last_weather

    if not WEATHER_KEY:
        return None

    lat, lon = get_lat_lon()

    url = "http://api.weatherapi.com/v1/current.json"
    params = {"key": WEATHER_KEY, "q": f"{lat},{lon}", "aqi": "no"}

    resp = requests.get(url, params=params, timeout=5)
    data = resp.json()

    # fail-safe: if API returns an error payload
    if "error" in data:
        # Return None
        return None

    temp_f = data["current"]["temp_f"]
    condition = data["current"]["condition"]["text"]
    loc_name = data["location"]["name"]

    last_weather = (temp_f, condition, loc_name)
    last_weather_time = now
    return last_weather


#HAND TRACKING HELPERS
def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def hit_widget(wgt, cx, cy):
    return (wgt["x"] <= cx <= wgt["x"] + wgt["w"] and
            wgt["y"] <= cy <= wgt["y"] + wgt["h"])


#TEXT / UI DRAW HELPERS
def draw_wrapped_text(canvas, text, x, y, max_width, font, scale, color, thickness, line_gap=22):
    #Draw text wrapped within max_width; returns new y after drawing.
    words = text.split()
    line = ""
    line_y = y

    for word in words:
        test_line = (line + " " + word).strip()
        (line_width, _), _ = cv2.getTextSize(test_line, font, scale, thickness)

        if line_width > max_width and line:
            cv2.putText(canvas, line, (x, line_y), font, scale, color, thickness)
            line = word
            line_y += line_gap
        else:
            line = test_line

    if line:
        cv2.putText(canvas, line, (x, line_y), font, scale, color, thickness)
        line_y += line_gap

    return line_y


def rounded_rect(img, x, y, w, h, r, color, thickness=2):
    # corners
    cv2.circle(img, (x + r, y + r), r, color, thickness)
    cv2.circle(img, (x + w - r, y + r), r, color, thickness)
    cv2.circle(img, (x + r, y + h - r), r, color, thickness)
    cv2.circle(img, (x + w - r, y + h - r), r, color, thickness)

    # edges
    cv2.line(img, (x + r, y), (x + w - r, y), color, thickness)
    cv2.line(img, (x + r, y + h), (x + w - r, y + h), color, thickness)
    cv2.line(img, (x, y + r), (x, y + h - r), color, thickness)
    cv2.line(img, (x + w, y + r), (x + w, y + h - r), color, thickness)


def draw_widget_card(canvas, wgt, grabbed=False):
    #Draw rounded 'card' with optional glow when grabbed
    x, y, w, h = wgt["x"], wgt["y"], wgt["w"], wgt["h"]
    r = 14

    fill = UI["card_fill"]
    border = UI["accent"] if grabbed else UI["border"]

    # fill (rounded)
    cv2.rectangle(canvas, (x + r, y), (x + w - r, y + h), fill, -1)
    cv2.rectangle(canvas, (x, y + r), (x + w, y + h - r), fill, -1)
    cv2.circle(canvas, (x + r, y + r), r, fill, -1)
    cv2.circle(canvas, (x + w - r, y + r), r, fill, -1)
    cv2.circle(canvas, (x + r, y + h - r), r, fill, -1)
    cv2.circle(canvas, (x + w - r, y + h - r), r, fill, -1)

    # glow under border
    if grabbed:
        rounded_rect(canvas, x, y, w, h, r, UI["accent"], thickness=6)

    # border
    rounded_rect(canvas, x, y, w, h, r, border, thickness=2)

    # title
    title = wgt["id"].upper()
    cv2.putText(canvas, title, (x + PAD_X, y + TITLE_Y_OFFSET), FONT, 0.7, UI["text"], 2)


#WIDGET CONTENT DRAWERS
def draw_clock(canvas, wgt):
    x, y, w, h = wgt["x"], wgt["y"], wgt["w"], wgt["h"]
    cx = x + PAD_X
    cy = y + CONTENT_TOP_OFFSET

    now = datetime.now()
    time_str = now.strftime("%I:%M %p")
    date_str = now.strftime("%a, %b %d")

    cv2.putText(canvas, time_str, (cx, cy + 20), FONT, 1.05, wgt["color"], 2)
    cv2.putText(canvas, date_str, (cx, cy + 50), FONT, 0.7, UI["muted"], 2)


def draw_notes(canvas, wgt):
    x, y, w, h = wgt["x"], wgt["y"], wgt["w"], wgt["h"]
    cx = x + PAD_X
    cy = y + CONTENT_TOP_OFFSET

    lines = wgt.get("text", "").split("\n")
    line_y = cy + 10

    for line in lines:
        if line_y > y + h - BOTTOM_PAD:
            break
        cv2.putText(canvas, line, (cx, line_y), FONT, 0.6, wgt["color"], 2)
        line_y += 24


def draw_news(canvas, wgt, headlines):
    x, y, w, h = wgt["x"], wgt["y"], wgt["w"], wgt["h"]
    cx = x + PAD_X
    cy = y + CONTENT_TOP_OFFSET

    max_text_width = w - (PAD_X * 2)
    line_y = cy + 10

    for headline in headlines[:5]:
        short = headline
        if len(short) > 140:
            short = short[:137] + "..."

        line_y = draw_wrapped_text(
            canvas, short,
            cx, line_y,
            max_text_width,
            FONT, 0.58,
            wgt["color"], 2,
            line_gap=20
        )

        if line_y > y + h - 28:
            break

    # footer
    footer_y = y + h - BOTTOM_PAD
    cv2.putText(canvas, "Source: BBC World", (cx, footer_y), FONT, 0.5, UI["muted"], 1)


def draw_weather(canvas, wgt):
    x, y, w, h = wgt["x"], wgt["y"], wgt["w"], wgt["h"]
    cx = x + PAD_X
    cy = y + CONTENT_TOP_OFFSET

    weather = get_weather()
    if not weather:
        cv2.putText(canvas, "Weather unavailable", (cx, cy + 20), FONT, 0.6, UI["muted"], 2)
        return

    temp_f, condition, loc_name = weather

    # location
    cv2.putText(canvas, loc_name, (cx, cy + 15), FONT, 0.65, wgt["color"], 2)

    # temp + fake degree
    temp_str = f"{temp_f:.1f}"
    cv2.putText(canvas, temp_str, (cx, cy + 45), FONT, 0.85, wgt["color"], 2)

    # fake degree symbol placed relative to temp text width
    (tw, th), _ = cv2.getTextSize(temp_str, FONT, 0.85, 2)
    deg_x = cx + tw + 6
    deg_y = cy + 33
    cv2.circle(canvas, (deg_x, deg_y), 4, wgt["color"], 2)
    cv2.putText(canvas, "F", (deg_x + 10, cy + 45), FONT, 0.85, wgt["color"], 2)

    # condition (wrapped)
    max_text_width = w - (PAD_X * 2)
    draw_wrapped_text(canvas, condition, cx, cy + 75, max_text_width, FONT, 0.58, UI["muted"], 2, line_gap=20)

#MAIN

def main():
    # Screen size (use last monitor or only one)
    screen_w = None
    screen_h = None
    for monitor in get_monitors():
        screen_w, screen_h = monitor.width, monitor.height

    # Camera + Mediapipe
    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(0)

    cv2.namedWindow("SmartGlass", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("SmartGlass", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    alpha = 0.25
    smooth_x, smooth_y = 0, 0
    cursor_x, cursor_y = 0, 0

    # News headlines
    url = "https://feeds.bbci.co.uk/news/world/rss.xml"
    feed = feedparser.parse(url)
    headlines = [str(entry.title) for entry in feed.entries[:5]]

    # Widgets
    widgets = [
        {"id": "clock", "x": 200, "y": 150, "w": 260, "h": 110, "color": (255, 255, 255)},
        {"id": "notes", "x": 600, "y": 250, "w": 320, "h": 170, "color": (200, 255, 200),
         "text": "- Finish CS homework\n- Test smart glass\n- Email Ofa"},
        {"id": "news", "x": 100, "y": 350, "w": 560, "h": 260, "color": (255, 200, 255)},
        {"id": "weather", "x": 900, "y": 150, "w": 320, "h": 160, "color": (200, 255, 255)},
    ]

    dragging_id = None
    grab_offset_x, grab_offset_y = 0, 0

    with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
    ) as hands:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h_cam, w_cam, _ = frame.shape

            # Canvas background
            canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            canvas[:] = UI["bg"]

            # Draw widget cards + content
            for wgt in widgets:
                grabbed = (dragging_id == wgt["id"])
                draw_widget_card(canvas, wgt, grabbed=grabbed)

                if wgt["id"] == "clock":
                    draw_clock(canvas, wgt)
                elif wgt["id"] == "notes":
                    draw_notes(canvas, wgt)
                elif wgt["id"] == "news":
                    draw_news(canvas, wgt, headlines)
                elif wgt["id"] == "weather":
                    draw_weather(canvas, wgt)

            # Hand tracking
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            fist = False

            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]

                wrist = hand.landmark[0]
                thumb_tip = hand.landmark[4]
                index_tip = hand.landmark[8]
                middle_tip = hand.landmark[12]
                ring_tip = hand.landmark[16]
                pinky_tip = hand.landmark[20]

                # map cursor to screen with smoothing
                target_x = int(index_tip.x * screen_w)
                target_y = int(index_tip.y * screen_h)

                smooth_x = int(alpha * target_x + (1 - alpha) * smooth_x)
                smooth_y = int(alpha * target_y + (1 - alpha) * smooth_y)
                cursor_x, cursor_y = smooth_x, smooth_y

                # 3 fingers close detection
                d_index = dist(index_tip, wrist)
                d_middle = dist(middle_tip, wrist)
                d_ring = dist(ring_tip, wrist)
                d_pinky = dist(pinky_tip, wrist)
                d_thumb = dist(thumb_tip, wrist)

                threshold = 0.15
                close_count = 0
                for d in (d_index, d_middle, d_ring, d_pinky, d_thumb):
                    if d < threshold:
                        close_count += 1
                # look like fist
                fist = (close_count >= 3)

                # drag widgets
                if fist:
                    if dragging_id is None:
                        for wgt in widgets:
                            if hit_widget(wgt, cursor_x, cursor_y):
                                dragging_id = wgt["id"]
                                grab_offset_x = cursor_x - wgt["x"]
                                grab_offset_y = cursor_y - wgt["y"]
                                break
                    else:
                        for wgt in widgets:
                            if wgt["id"] == dragging_id:
                                new_x = cursor_x - grab_offset_x
                                new_y = cursor_y - grab_offset_y

                                new_x = max(0, min(screen_w - wgt["w"], new_x))
                                new_y = max(0, min(screen_h - wgt["h"], new_y))

                                wgt["x"], wgt["y"] = new_x, new_y
                                break
                else:
                    dragging_id = None

            # highlight + size when grabbing
            cursor_color = CURSOR_GRAB_COLOR if fist else CURSOR_IDLE_COLOR
            cursor_r = CURSOR_GRAB_R if fist else CURSOR_IDLE_R
            cv2.circle(canvas, (cursor_x, cursor_y), cursor_r, cursor_color, -1)

            # show UI
            cv2.imshow("SmartGlass", canvas)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
