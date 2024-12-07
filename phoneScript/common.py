from appium.webdriver.common.appiumby import By


def tryFind(driver, by=By.XPATH, value=''):
    global ele
    try:
        ele = driver.find_element(by, value)
    except:
        print('error ...')
        return None
    return ele


def click(driver, xpath):
    btn = tryFind(driver, By.XPATH, xpath)
    if btn is None:
        return 0
    print('ok ...')
    btn.click()
    return 1


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


def swipe_up(driver, top=0, bottom=1):
    s = driver.get_window_size()
    height = s['height']
    width = s['width']
    length = height*(bottom - top)
    x = width * 0.5  # x坐标
    y1 = height * top + length * 0.7  # 起点y坐标
    y2 = height * top + length * 0.3  # 终点y坐标
    driver.swipe(x, y1, x, y2, 15)