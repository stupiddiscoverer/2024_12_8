import os
import subprocess


def check_video(file_path):
    if not file_path.endswith('.mp4'):
        return
    try:
        # 通过 ffprobe 获取视频文件信息
        result = subprocess.run(
            ['ffprobe', file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        # 检查返回值
        if result.returncode != 0:
            print(f"{file_path} is corrupted: {result.stderr}")
            os.remove(file_path)
        else:
            print(f"{file_path} is valid.")
    except FileNotFoundError:
        print("ffprobe is not installed or not found in PATH.")


def check_path(folder_path):
    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            check_video(file_path)


# 替换为你想检查的视频文件路径
check_path("C:/Users/张三/Videos")
