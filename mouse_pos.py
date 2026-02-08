import pyautogui
import time

print("Move mouse to target location. Press Ctrl+C to stop.")
while True:
    x, y = pyautogui.position()
    print(f"X={x}, Y={y}", end="\r")
    time.sleep(0.1)
