import os
import hashlib
import subprocess


def get_file_hash(file_path):
    """计算文件的 SHA-256 哈希值"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # 逐块读取文件
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def remove_duplicates(folder_path):
    """在给定文件夹中删除重复文件，仅保留一份"""
    file_lens = []
    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            file_len = os.path.getsize(file_path)
            file_lens.append((file_len, file_path))

    file_lens.sort(key=lambda x: x[0])

    old_len = file_lens[0][0]
    old_hash = 0
    for index in range(1, len(file_lens)):
        file = file_lens[index]
        if file[0] == old_len:
            if old_hash == 0:
                old_hash = get_file_hash(file_lens[index - 1][1])
            this_hash = get_file_hash(file[1])
            if this_hash == old_hash:
                print('delete dup file %s, size %d' % (file[1], file[0]))
                os.remove(file[1])
        else:
            old_hash = 0
            old_len = file[0]


def removeFileByRe(path='/', patt=''):
    path = path.replace("\\", "/")
    if not path.endswith('/'):
        path += '/'
    subprocess.run("del %s%s" % (path, patt))


if __name__ == '__main__':
    remove_duplicates('C:/Users/张三/Videos')
    # removeFileByRe('C:/Users/张三/Videos/xx', '*探花*')