import os
import sys

from appium import webdriver
import time
from appium.webdriver.common.appiumby import By
from selenium.webdriver import ActionChains
# from appium.webdriver.common.touch_action import TouchAction

capabilities = dict(
    platformName='Android',
    # platformVersion='14',
    automationName='uiautomator2',
    # deviceName='192.168.2.13:5555',
    deviceName='Android',
    # appPackage='com.immomo.vchat',
    # appActivity='com.immomo.vchat.activity.HomeActivity',
    unicodeKeyboard=True,
    resetKeyboard=True,
    noReset=True
)
# 打开抖音并滑动成功过一次，不知道为啥大多数时候不成功，ok，必须先清理应用，不能在后台已经运行

driver = webdriver.Remote('http://localhost:4723', capabilities)
width = driver.get_window_size().get('width')
height = driver.get_window_size().get('height')
action = ActionChains(driver, duration=1000)
print('width:', width, '  height:', height)
time.sleep(1)


def tryFind(by, value):
    global ele
    try:
        ele = driver.find_element(by, value)
    except:
        print('error ...')
        return None
    return ele


def click(xpath):
    btn = tryFind(By.XPATH, xpath)
    if btn is None:
        return 0
    print('ok ...')
    try:
        btn.click()
    except:
        print('click failed, press return and retry')
        driver.press_keycode(keycode=4)  # 返回
        time.sleep(2)
        btn.click()
    time.sleep(1)
    return 1


def sendmsg(msg):
    ele = tryFind(By.XPATH, '//android.widget.EditText[@resource-id="com.immomo.vchat:id/et_input_message"]')
    if ele is None:
        return
    ele.send_keys(msg)
    if not click('//android.widget.TextView[@resource-id="com.immomo.vchat:id/tv_room_send"]'):
        return
    # if not click('(//android.widget.TextView[@resource-id="com.immomo.vchat:id/meet_again_pai_pai"])[1]'):
    #     return


def closeAd():
    if click('//android.widget.FrameLayout[@resource-id="com.immomo.vchat:id/loveMatchStartSuccessGoTv"]'):
        sendmsg('哈喽啊，公主')
        click('//android.widget.ImageView[@resource-id="com.immomo.vchat:id/iv_common_back"]')
    click('//android.widget.ImageView[@resource-id="com.immomo.vchat:id/room_activity_banner_close_icon"]')
    click('//android.widget.TextView[@resource-id="com.immomo.vchat:id/btn_close"]')
    click('//android.widget.ImageView[@resource-id="com.immomo.vchat:id/room_recommend_join_close"]')


def swipe(xpath, num1, num2):
    ele = tryFind(By.XPATH, xpath + '[' + str(num1) + ']')
    if ele is None:
        return False
    action.w3c_actions.pointer_action.move_to(ele).pointer_down().pause(1)
    ele = tryFind(By.XPATH, xpath + '[' + str(num2) + ']')
    action.w3c_actions.pointer_action.move_to(ele).pause(1).pointer_up()
    action.perform()
    return True


def myFocus():
    closeAd()
    if not click('//android.widget.TextView[@text="我的"]'):
        return
    closeAd()
    ele = tryFind(By.XPATH, value='//android.widget.TextView[@resource-id="com.immomo.vchat:id/new_profile_follow_count_tv"]')
    if ele is None:
        return
    focusCount = int(ele.text)
    print('focusCount: ', focusCount)
    if not click('//android.widget.LinearLayout[@resource-id="com.immomo.vchat:id/ll_follow_container"]'):
        return
    for j in range(focusCount//8):
        time.sleep(1)
        for i in range(1, 9):
            time.sleep(1)
            ele = tryFind(By.XPATH, '(//android.widget.TextView[@resource-id="com.immomo.vchat:id/tv_relation_button"])[' + str(i) + ']')
            if ele is None:
                print('error line: ', str(sys._getframe().f_lineno))
                return
            if ele.text == '邀请回关':
                ele.click()
                time.sleep(1)
                click('//android.widget.TextView[@resource-id="com.immomo.vchat:id/gift_invite_follow_panel_close_chat"]')
                ele = tryFind(By.XPATH,
                              '(//android.widget.TextView[@resource-id="com.immomo.vchat:id/tv_relation_button"])[' + str(
                                  i) + ']')
            ele.click()
            time.sleep(1)
            sendmsg('哈喽啊，公主')
            click('//android.widget.ImageView[@resource-id="com.immomo.vchat:id/iv_common_back"]')
        time.sleep(1)
        if not swipe('(//android.widget.LinearLayout[@resource-id="com.immomo.vchat:id/ll_user_info_container"])',
                     9, 1):
            return
    driver.press_keycode(keycode=4)  # 返回
    print('end of focus')


def reMeet():
    closeAd()
    if not click('//android.widget.TextView[@text="消息"]'):
        return
    closeAd()
    if not click('//android.widget.TextView[@text="再相遇"]'):
        return
    for i in range(8):
        time.sleep(1)
        if not click('(//android.widget.TextView[@resource-id="com.immomo.vchat:id/meet_again_pai_pai"])[1]'):
            return
    driver.press_keycode(keycode=4)  # 返回
    print('end of reMeet')


def parseInt(s):
    a = 0
    readed = False
    start = False
    for c in s:
        if not readed and '0' <= c <= '9':
            start = True
            a = a*10 + ord(c) - ord('0')
        else:
            if start:
                readed = True
    return a


def pipei():
    # 匹配
    closeAd()
    if not click('//android.widget.TextView[@resource-id="com.immomo.vchat:id/tv_home_tab_desc" and @text="找朋友"]'):
        return
    closeAd()
    ele = tryFind(By.XPATH, '//android.widget.TextView[@resource-id="com.immomo.vchat:id/tv_one_des"]')
    if ele is None:
        return
    chance = parseInt(ele.text)
    print('chance: ', chance)
    for i in range(chance):
        closeAd()
        click('//android.view.ViewGroup[@resource-id="com.immomo.vchat:id/top_one"]')
        time.sleep(7)
        sendmsg('公主做我女朋友')
        time.sleep(1)
        click('//android.widget.ImageView[@resource-id="com.immomo.vchat:id/iv_common_back"]')
    print('end of pipei')
    # driver.press_keycode(keycode=4)  # 返回


def findFriends():
    closeAd()
    if not click('//android.widget.TextView[@resource-id="com.immomo.vchat:id/tv_home_tab_desc" and @text="找朋友"]'):
        return
    closeAd()
    if not click('(//android.widget.TextView[@text="找朋友"])[1]'):
        return
    for i in range(1000):
        eles = driver.find_elements(by=By.XPATH, value='(//android.widget.FrameLayout[@resource-id="com.immomo.vchat:id/home_avatar_layout"])')
        if eles is None:
            return
        num = len(eles)
        for i in range(1, num+1):
            if not click('(//android.widget.FrameLayout[@resource-id="com.immomo.vchat:id/home_avatar_layout"])[' + str(i) +']'):
                return
            ele = tryFind(By.XPATH, '//android.widget.TextView[@resource-id="com.immomo.vchat:id/room_owner_name"]')
            if ele is not None:
                click('//android.widget.ImageView[@resource-id="com.immomo.vchat:id/iv_change_room_function"]')
                click('//android.widget.TextView[@resource-id="com.immomo.vchat:id/tv_onwer_panel_name" and @text="退出房间"]')
                time.sleep(1)
                continue
            click('//android.widget.TextView[@text="聊天"]')
            sendmsg('如果你需要182的男朋友，请考虑我')
            driver.press_keycode(keycode=4)  # 返回
            time.sleep(1)
            driver.press_keycode(keycode=4)  # 返回
        swipe('(//android.widget.FrameLayout[@resource-id="com.immomo.vchat:id/home_avatar_layout"])', num, 1)
        time.sleep(1)




def startHz():
    print('start')
    pipei()
    reMeet()
    findFriends()
    myFocus()
    driver.quit()


def test():
    tryFind(By.XPATH, '//android.widget.EditText[@resource-id="com.android.chrome:id/url_bar"]').send_keys(u'你好sd')
    action.perform()
        # time.sleep(1)


if __name__ == '__main__':
    # test()
    # myFocus()
    # os.system('adb devices')
    print(os.popen('adb devices').read())
    startHz()
    # findFriends()
