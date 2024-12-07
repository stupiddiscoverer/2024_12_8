import pyautogui
import time


def press_key(key, duration=0.1):
    pyautogui.keyDown(key)
    time.sleep(duration)
    pyautogui.keyUp(key)


def perform_moves():
    # 等待游戏开始
    time.sleep(3)

    while True:
        # 示例操作：向前移动
        press_key('right', 0.5)
        time.sleep(0.5)

        # 示例操作：跳跃
        press_key('up', 0.5)
        time.sleep(0.5)

        # 示例操作：攻击
        press_key('a', 0.1)  # 假设 'a' 是轻拳
        time.sleep(0.5)

        # 停止循环以避免无限循环
        break


if __name__ == "__main__":
    perform_moves()
