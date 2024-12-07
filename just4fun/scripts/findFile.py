import os
import re
import sys


class SingleNode:
    def __init__(self, data=None, next=None, dir=None):
        self.data = data
        self.next = next
        self.dir = dir


def ispathchar(c):
    if '!' <= c <= '~':
        return True
    if c > 256:
        return True


def getfilename(filename1):
    filename1 += ''
    filename1 = filename1.replace('\\', '/')
    if filename1.endswith('/'):
        filename1 = filename1[0:-1]
    lens = len(filename1) - 1
    while filename1[lens] != '/':
        lens -= 1
    return filename1[lens:]


def isspace(c):
    return c == ' ' or c == '\n' or c == '\r' or c == '\t'


def trim(string):
    start = 0
    end = len(string) - 1
    while start < end and isspace(string[start]):
        start += 1
    while end >= 0 and isspace(string[end]):
        end -= 1
    return string[start:end + 1]


def getpath(path):
    path += ''
    path = trim(path)
    path = path.replace('\\', '/')
    index = 0
    while index < len(path) and ispathchar(path[index]) and path[index] != '/' and path[index] != ':':
        index += 1
    if index == len(path):
        return ''
    if path[index] != ':':
        return ''
    if path[index + 1] != '/':
        return ''

    if path.endswith('/'):
        return path
    path += '/'
    if os.path.isdir(path):
        return path
    end = len(path) - 1
    while path[end] != '/':
        end -= 1
    return path[0:end + 1]


def findFile(path, filename):
    files = os.listdir(path)

    tempNode = SingleNode(data=files, next=None, dir=path)
    endNode = tempNode

    while tempNode is not None:
        files = tempNode.data
        dir = tempNode.dir
        for file in files:
            # if not os.access(dir + file, os.R_OK):
            #     continue
            if re.fullmatch(filename, file):
                print((dir + file))
            if os.path.isdir(dir + file):
                try:
                    endNode.next = SingleNode(data=os.listdir(dir + file), next=None, dir=dir + file + '/')
                except IOError:
                    continue
                endNode = endNode.next
        tempNode = tempNode.next


# python转exe：pip install pyinstaller  // Pyinstaller -F -w test.py
# -F参数代表制作独立的可执行程序。
# -w是指程序启动的时候不会打开命令行。
if __name__ == '__main__':
    dir = os.getcwd().replace('\\', '/')
    if 1 < len(sys.argv) < 4:
        if len(sys.argv) == 3:
            dir = sys.argv[1]
        if not os.path.exists(dir):
            print(dir, "is not a directory")
        else:
            if not dir.endswith('/'):
                dir += '/'
            filename = sys.argv[-1]
            findFile(dir, filename)
    else:
        print('current dir is:', dir)
        print('usage: findFile a.*b 查找当前目录下 a开头b结尾的文件，')
        print('usage: findFile dirA a.*b 查找目录dirA下 a开头b结尾的文件 dirA可以是相对目录')
        print('.*是python正则表达式，表示任意数量（包括0个）的任意字符，也可以用其他规则匹配')