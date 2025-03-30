import pydirectinput
import pyautogui
import time

while True:
    time.sleep(10)              # 10 秒后再按一次
    pydirectinput.keyDown("w")  # 按住 "W"
    pydirectinput.keyUp("w")    # 释放 "W"ww
    time.sleep(1)          # 按住 0.1 秒
    pyautogui.keyDown("w")  # 按住 "W"
    pyautogui.keyUp("w")    # 释放 "W"


