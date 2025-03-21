
def getDirectory(filePath=''):
    if filePath.__contains__('\\'):
        filePath = filePath.replace('\\', '/')
        return filePath[0:filePath.rindex('/')+1]


if __name__ == '__main__':
    print(getDirectory('C:\\Users\张三\Pictures\IMG_20231114_092847.jpg'))