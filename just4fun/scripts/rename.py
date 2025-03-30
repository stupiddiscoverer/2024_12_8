import os
import re
import shutil

# 获取当前目录
current_directory = 'C:\\Users\\张三\\Music\\7.怀旧情怀金典200首\\'
current_directory1 = 'C:\\Users\\张三\\Music\\1抖音流行热歌精选榜900首\\'
# 正则表达式：匹配以数字和点开头的部分（如 001. 或 101.）
pattern = re.compile(r'^\d+\.\s*')
pattern1 = re.compile(r'.*-')

nameset = set()
files = os.listdir(current_directory1)
for file in files:
    if file.endswith('.lrc'):
        nameset.add(file[0:-4])
for file in files:
    if file.endswith('.mp3'):
        if file[0:-4] not in nameset:
            print(f'Move-Item -Path "{current_directory1+file}" -Destination "{current_directory+file}"')


# 遍历目录中的所有文件
for filename in os.listdir(current_directory):
    if filename.endswith('.lrc'):
        file = filename[0:-4]
        print(f'Move-Item -Path "{current_directory+file}.lrc" -Destination "{current_directory1+file}.lrc"')
        print(f'Move-Item -Path "{current_directory+file}.mp3" -Destination "{current_directory1+file}.mp3"')
    # 构造新文件名
    new_filename = filename.replace('【AA热播】A', '').replace(
        '【电子】', '').replace('【节奏热播】', '').replace(
        '【英文蓝调R&b】', '').replace('【AA热播】a', '')
    if pattern.match(new_filename):
        new_filename = pattern.sub('', filename)
    if '-' in new_filename:  # 如果文件名包含'-'
        # 获取'-'后面的部分
        new_filename = filename.split('-', 1)[-1]

    if new_filename != filename:
        # 获取完整的文件路径
        old_file = os.path.join(current_directory, filename)
        new_file = os.path.join(current_directory, new_filename)
        if not os.path.exists(new_file):
            # 重命名文件
            os.rename(old_file, new_file)
            print(f'Renamed: {filename} -> {new_filename}')
