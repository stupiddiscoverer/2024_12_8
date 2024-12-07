import os.path
import re
import subprocess

import requests
from urllib.parse import unquote

header = {
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'accept-encoding': 'gzip, deflate, br, zstd',
    'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6,zh-TW;q=0.5',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36 Edg/127.0.0.0'
}


def download(url):
    return requests.get(url, headers=header)

def tmpUrp(url=''):
    for i in range(2, len(url)):
        if url[-i] == '/':
            return url[0:-i+1]


def getDirStr(url):
    return url[-21:-11]


def downloadTsAndSave(segmts=[], method='', uri='', iv=''):
    tempDir = getDirStr(m3u8url)
    key = download(uri).text
    print(key)
    if len(key) < 32:
        key = '0'*(32 - len(key)) + key
    if iv.startswith('0x'):
        iv = iv[2:]
    if not os.path.exists(tempDir):
        os.makedirs(tempDir)
    for ts in segmts:
        resp = requests.get(tmpurp+ts, headers=header)
        with open(tempDir+'/'+ts, "wb") as f:
            f.write(resp.content)
            command = "ffmpeg -i %s -c copy -decryption_key %s -decryption_iv %s %s" % (tempDir+'/'+ts, key,
                                                    iv, tempDir+'/'+ts.replace(".ts", '.mp4'))
            print(command)
            subprocess.run(command)
        return

def downloadAndDecodeTs(text):
    lines = text.split('\n')
    segmts = []
    method = ''
    uri = ''
    iv = ''
    for line in lines:
        if 'EXT-X-KEY' in line:
            method = re.findall('METHOD=.*', text)[0][7:]
            print(method)
            uri = re.findall('[Uu][Rr][Ii]=["\'].*["\']', text)[0][5:-1]
            if not uri.startswith('http'):
                uri = tmpurp + uri
            print(uri)
            iv = re.findall('IV=.*', text)[0][3:]
            print(iv)
        if line.endswith('.ts'):
            segmts.append(line)
    downloadTsAndSave(segmts, method, uri, iv)


if __name__ == '__main__':
    m3u8url = 'https://res.cuieyi.com/m3u8/2024-06-12/001/d00798579373bf1a/index.m3u8'
    print(getDirStr(m3u8url))
    tmpurp = tmpUrp(m3u8url)
    print(tmpurp)
    m3u8Text = unquote(requests.get(m3u8url, headers=header).text)
    downloadAndDecodeTs(m3u8Text)

