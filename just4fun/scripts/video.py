import os
import hashlib
import subprocess
import ffmpeg
from mutagen.mp4 import MP4
from moviepy import VideoFileClip, concatenate_videoclips


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


def remove_disorder(folder_path):
    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            check_video(file_path)


def mergeVideo(video_folder='', outPath=''):
    # 获取所有 MP4 视频文件
    video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]

    # 按 "标题 + 分辨率 + 帧率" 进行分组
    video_groups = {}

    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)

        # 获取 MP4 元数据中的 "标题"（Title）
        try:
            metadata = MP4(video_path).tags
            title = metadata.get("\xa9nam", ["Unknown"])[0]  # 获取标题（默认值 "Unknown"）
        except Exception as e:
            print(f"无法获取 {video_file} 的标题: {e}")
            continue
        if title == 'Unknown' or title == 'na' or title == 'unknown':
            continue
        print(title)

        # 获取视频分辨率和帧率
        probe = ffmpeg.probe(video_path, v='error', select_streams='v:0',
                             show_entries='stream=width,height,r_frame_rate')
        resolution = (probe['streams'][0]['width'], probe['streams'][0]['height'])
        frame_rate = eval(probe['streams'][0]['r_frame_rate'])  # 转换为浮点数

        # 以 (标题, 分辨率, 帧率) 为 key 进行分组
        video_key = (title, resolution, frame_rate)

        if video_key not in video_groups:
            video_groups[video_key] = []

        video_groups[video_key].append(video_path)

    # 合并同组视频
    for (title, resolution, frame_rate), videos in video_groups.items():
        if len(videos) > 1:
            print(f"合并以下视频（标题: {title}, 分辨率: {resolution}, 帧率: {frame_rate}）：")
            for v in videos:
                print(f"  - {v}")
            # 读取视频
            clips = [VideoFileClip(video) for video in videos]
            # 合并视频
            final_clip = concatenate_videoclips(clips, method="compose")
            # 以第一个视频的文件名作为输出文件名
            output_file = os.path.join(outPath, os.path.basename(videos[0]))
            # 保存合并后的视频
            final_clip.write_videofile(output_file, codec="libx264", fps=frame_rate)
            # 释放资源
            final_clip.close()
            for clip in clips:
                clip.close()
        else:
            print(f"标题 {title} 只有一个视频，无需合并。")


if __name__ == '__main__':
    # remove_duplicates('C:/Users/张三/Videos')
    # 替换为你想检查的视频文件路径
    # remove_disorder("C:/Users/张三/Videos")
    # 设置视频文件夹路径
    video_folder = "d:/video/vid"
    outPath = "d:/test/"
    mergeVideo(video_folder, outPath)
    # removeFileByRe('C:/Users/张三/Videos/xx', '*探花*')