# https://res.cuieyi.com/m3u8/2024-08-05/minio/591a11bc2afaa36e/index.m3u8
import os.path
import random
import re
import subprocess
import threading
import time
from urllib.parse import unquote

import psutil
import requests
from lxml import html

# print(time.localtime(1723046095))

baseUrl = "https://3.xiu2049a.cc:8888"
# baseUrl = 'https://6.xx1552.cc:8888'

header = {
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'accept-encoding': 'gzip, deflate, br, zstd',
    'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6,zh-TW;q=0.5',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36 Edg/127.0.0.0'
}

maxThread = 3
threadCount = 0
lock = threading.Lock()
pageNum = 1
pageEnd = 50


class vidInfo:
    def __init__(self, url, title):
        self.url = url
        self.title = title


class ring:
    def __init__(self, len):
        assert len > 1
        self.len = len
        self.list = [''] * len
        self.index = 0      # 等待插入的下标
        self.getIndex = 0   # 等待获取的下标，如果和等待插入下标相等代表ring空了

    def add(self, ele):
        if (self.index + 1) % self.len == self.getIndex:
            return False
        self.list[self.index] = ele
        self.index = (self.index+1) % self.len

    def get(self) -> vidInfo:
        if self.index == self.getIndex:
            return None
        ele = self.list[self.getIndex]
        self.getIndex = (self.getIndex+1) % self.len
        return ele

    def remainLen(self):
        if self.index > self.getIndex:
            return self.index - self.getIndex
        return self.getIndex - self.index


infoRing = ring(16)


def isNetSpeedOk(interface='WLAN'):
    pre = psutil.net_io_counters(pernic=True)[interface].bytes_recv
    time.sleep(3)
    if (psutil.net_io_counters(pernic=True)[interface].bytes_recv - pre) > 1024 * 1024:  # 3秒下载量要大于1MB
        return True
    return False


def downloadVid(url, title):
    global threadCount, lock
    with lock:
        threadCount += 1
    url = baseUrl + url
    print(url, title)
    path = 'C:/Users/张三/Videos/xx/' + title + '.mp4'
    if os.path.exists(path):
        time.sleep(3)
        with lock:
            threadCount -= 1
        return
    time.sleep(random.random() * 4)
    old = time.time()
    resp = requests.get(url, headers=header)
    text = unquote(resp.text)
    pattern = r'http.*?index\.m3u8'
    matches = re.findall(pattern, text)
    if len(matches) < 1:
        with lock:
            threadCount -= 1
        return
    print(matches[0], title)
    cpu = 'libx264'
    gpu = 'h264_nvenc'
    cmd = ('ffmpeg  '
           '-reconnect_streamed 1 '
           '-headers "accept-encoding: gzip, deflate, br, zstd\\r\\n '
           'accept-language: zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6,zh-TW;q=0.5\\r\\n '
           # 'origin: https://6.xx1466.cc:8888\\r\\n '
           'referer: ' + baseUrl + '/\\r\\n '
           'user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
           'Chrome/127.0.0.0 Safari/537.36 Edg/127.0.0.0\\r\\n" '
           # '-hwaccel nvdec -i "%s" -movflags +faststart -fflags +genpts -preset fast'
           #                         ' -c:v %s -c:a aac -f mp4 "%s.part.mp4"') % (matches[0], gpu, path)
           '-i "%s" -c copy "%s.part.mp4"') % (matches[0], path)
    ##'ffmpeg',  '-i', m3u8_file,
    # 启动一个进程
    try:
        process = subprocess.Popen(cmd)
    except:
        with lock:
            threadCount -= 1
        return
    # 获取进程ID
    success = False
    pre_read = 0
    for i in range(30*6):
        time.sleep(10)
        if process.poll() is not None:
            success = True
            break
        else:
            curRead = psutil.Process(process.pid).io_counters().write_bytes
            if curRead - pre_read < 100*1024:  #10秒下载量超过100KB才行
                print(f'download too slow ... {curRead, pre_read, title}')
                break
            pre_read = curRead
    # result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if success:
        if process.returncode == 0:
            os.rename(path + '.part.mp4', path)
            print(" using time: %d min %d sec" % ((time.time() - old) / 60, (time.time() - old) % 60),
                  title, os.path.getsize(path) / 1024 // 1024, "MB")
    else:
        try:
            process.terminate()
            # process.send_signal(signal.SIGINT)
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pid = process.pid
            subprocess.run(['taskkill', '/PID', str(pid), '/F'])
            print(f"taskkill process with PID: {pid}")
    with lock:
        threadCount -= 1


def clean():
    result = subprocess.run('taskkill /IM "ffmpeg.exe" /F', shell=True, capture_output=True, text=True)
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)
    result = subprocess.run('del C:\\Users\\张三\\Videos\\xx\\*.part.mp4', shell=True, capture_output=True, text=True)
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)


def getLinks():
    global infoRing, pageNum, pageEnd
    if pageNum > pageEnd:
        quit()
    # print(baseUrl)
    resp = requests.get(baseUrl + '/category/30/' + str(pageNum), headers=header)
    text = unquote(resp.text)
    # print(text)
    dom = html.fromstring(text)
    links = dom.xpath("//div[@class='vod-item']/a[@class='vod-title']")
    print('page: ', pageNum, 'len：', len(links))
    if len(links) == 0:
        getNewestSite(dom)
    return links


def fillRing():
    global pageNum
    links = getLinks()
    print(links)
    if len(links) == 0:
        links = getLinks()
    for link in links:
        vidTitle = (link.text.strip().replace(" ", '').replace("\r", '')
                    .replace("\t", '').replace("\n", '').replace("?", "")
                    .replace("/", "").replace("\\", "").replace("&amp;", "&")
                    .replace(":", "：").replace("|", "").replace("&", ''))
        vidLink = link.xpath('@href')
        infoRing.add(vidInfo(vidLink[0], vidTitle))
    pageNum += 1


def downloadPage():
    while True:
        if infoRing.remainLen() == 0:
            fillRing()
        threads = []
        if threadCount < maxThread:
            info = infoRing.get()
            if info is None:
                break
            thread = threading.Thread(target=downloadVid, args=(info.url, info.title))
            threads.append(thread)
            thread.start()
        time.sleep(1)
        if threadCount < 1:
            print('end')
            break
        # stop_event = threading.Event()
        # for t in threads:
        #     t.join(15*60)
            # if t.is_alive():
            #     print("超时，准备停止线程...")
            #     while isNetSpeedOk():
            #         time.sleep(10)
            #         print('still downloading ...')
            #     clean()
            #     time.sleep(3)
            #     stop_event.set()
            #     time.sleep(3)
            #     break


def getHost(url=''):
    parts = url.split('/')
    return parts[0] + '//' + parts[2]


def getNewestSite(dom):
    global baseUrl
    links = dom.xpath('//*[@id="as"]/@href')[0]
    baseUrl = getHost(links)
    print('new site is', links)
    with open('xiuxiu.py', 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('baseUrl ='):
                lines[i] = 'baseUrl = "' + baseUrl + '"\n'
                break
    with open('xiuxiu.py', 'w') as f:
        f.writelines(lines)


if __name__ == '__main__':
    # downloadPage(1)
    clean()
    downloadPage()
    # downloadVid('a', 'a')
