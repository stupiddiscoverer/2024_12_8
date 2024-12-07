from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By

driverfile_path = r'C:\Users\张三\Desktop\PycharmProjects\dirvers\msedgedriver.exe'

edge_options = webdriver.EdgeOptions()
edge_options.use_chromium = True
# 屏蔽inforbar
edge_options.add_experimental_option('useAutomationExtension', False)
edge_options.add_experimental_option('excludeSwitches', ['enable-automation', 'enable-logging'])
# 创建driver
driver = webdriver.Edge(executable_path=driverfile_path, options=edge_options)

driver.get(r'https://cloud.huawei.com/home#/home')
sleep(5)

driver.get

elements = driver.find_elements(by=By.CSS_SELECTOR, value='input')
element = elements
print(element)
# element.click()

# sleep(5)
# driver.close()