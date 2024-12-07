import os
import subprocess
import threading
import time

from appium import webdriver
from appium.webdriver.common.appiumby import By
from selenium.webdriver import ActionChains

import common

capabilities = dict(
    platformName='Android',
    # platformVersion='14',
    automationName='uiautomator2',
    # deviceName='192.168.2.13:5555',
    deviceName='Android',
    appPackage='com.smile.gifmaker',
    appActivity='com.yxcorp.gifshow.HomeActivity',
    unicodeKeyboard=True,
    resetKeyboard=True,
    noReset=True
)


def swipe():
    ele = common.tryFind(driver, value='(//android.widget.ImageView[@content-desc="更多"])[3]')
    action.w3c_actions.pointer_action.move_to(ele).pointer_down()
    ele = common.tryFind(driver, value='(//android.widget.ImageView[@content-desc="更多"])[1]')
    action.w3c_actions.pointer_action.move_to(ele).pause(1).pointer_up()
    action.perform()


def commentVideo_x100_pro():
    print('x100pro start')
    common.click(driver, '//android.widget.CheckedTextView[@resource-id="android:id/text1" and @text="精选"]')
    action.w3c_actions.pointer_action.move_to_location(int(155), int(2682)).pointer_down().pointer_up()
    action.perform()
    time.sleep(1)
    driver.press_keycode(keycode=4)  # 返回
    driver.set_clipboard_text('我182，谁做我的女人😡')
    for i in range(100):
        print('start')
        # common.click(driver, '//android.widget.TextView[@resource-id="com.ss.android.ugc.aweme:id/lmn"]')
        common.swipe_up(driver)
        time.sleep(1)
        ele = common.tryFind(driver, by=By.CLASS_NAME,value='android.widget.TextSwitcher')
        if ele is None:
            continue
        action.w3c_actions.pointer_action.move_to_location(int(982/1080 * width), int(1510/2265 * height)).pointer_down().pointer_up()
        action.perform()
        action.w3c_actions.pointer_action.move_to_location(int(243/1080 * width), int(2324/2265 * height)).pointer_down().pointer_up()
        action.perform()
        # driver.press_keycode(keycode=67)  # 退格
        action.w3c_actions.pointer_action.move_to_location(535, 2320).pointer_down().pause(1).pointer_up()
        action.perform()
        action.w3c_actions.pointer_action.move_to_location(139, 2152).pointer_down().pointer_up()
        action.perform()
        action.w3c_actions.pointer_action.move_to_location(int(1135), int(2559)).pointer_down().pointer_up()
        action.perform()
        # time.sleep(0.5)
        action.w3c_actions.pointer_action.move_to_location(int(484/1080 * width), int(484/2265 * height)).pointer_down().pointer_up()
        action.perform()
        # driver.press_keycode(keycode=4)  # 返回,反应很慢
        print(i)
        # driver.press_keycode(keycode=4)  # 返回
    driver.press_keycode(keycode=4)  # 返回


def run_command(command):
    with subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
        # 循环读取标准输出
        for line in process.stdout:
            print("Output:", line.strip())

            # 如果有错误输出，也打印出来
        stderr_output = process.stderr.read().strip()
        if stderr_output:
            print("Error:", stderr_output)


if __name__ == '__main__':
    # thread = threading.Thread(target=run_command, args=('appium',))  # 最好自己开窗口运行appium
    # thread.start()
    driver = webdriver.Remote('http://localhost:4723', capabilities)
    width = driver.get_window_size().get('width')
    height = driver.get_window_size().get('height')
    print('width:', width, '  height:', height)
    time.sleep(1)
    action = ActionChains(driver, duration=800)

    print(os.popen('adb devices').read())
    if width == 1260:
        commentVideo_x100_pro()
    # common.swipe_up(driver)