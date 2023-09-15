import time
import cv2
import pyautogui
from gaze_tracking import GazeTracking


gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

avg_val, count, num = 0, 0, 0
max_x, min_x = float("-inf"), float("inf")
x_coords = set()


def scroll_down():
    """
    Scroll down function
    """
    pyautogui.press("down")
    time.sleep(0.2)


while True:
    _, frame = webcam.read()
    gaze.update(frame)

    l ,r = gaze.pupil_left_pos(), gaze.pupil_right_pos()
    if l and r:
        x_left, y_left = l
        x_right, y_right = r
        x = (x_left + x_right) / 2.0
        y = (y_left + y_right) / 2.0
        

        max_x, min_x = max(max_x, x), min(min_x, x)
        count += 1
        new_avg = x * 1.0 / count + avg_val * (1 - 1.0 / count)
        if new_avg > avg_val:
            num += 1
        else:
            num = 0
        avg_val = new_avg

        x_coords.add(x)
        if (num > 10) or (
            max_x - min_x >= 10 and x >= (max_x - min_x) * 0.75 + min_x
        ):
            avg_val, count, num = 0, 0, 0
            max_x, min_x = float("-inf"), float("inf")
            x_coords = set()
            scroll_down()


    if cv2.waitKey(1) == 27:
        break