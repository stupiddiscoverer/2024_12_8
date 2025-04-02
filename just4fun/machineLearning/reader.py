from gtts import gTTS
import os
import pygame


class NovelReader:
    def __init__(self, file_path):
        # 初始化pygame音频播放
        pygame.mixer.init()

        # 读取小说文件内容
        self.load_file(file_path)

    def load_file(self, file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                self.text_content = file.readlines()  # 读取为行
            print(f"文件内容已加载，开始逐行朗读...")
            self.start_reading()
        except Exception as e:
            print(f"加载文件时出错: {e}")

    def start_reading(self):
        # 逐行朗读文件
        for line in self.text_content:
            line = line.strip()  # 去掉行首尾空格
            if line:  # 如果行不是空行
                print(f"正在朗读: {line}")  # 打印当前朗读的行
                tts = gTTS(text=line, lang='zh')  # 设置中文语言
                tts.save("temp.mp3")  # 保存为临时文件

                # 使用pygame播放音频文件
                pygame.mixer.music.load("temp.mp3")
                pygame.mixer.music.play()

                # 等待音频播放完毕
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)


# 输入小说文件路径
file_path = "C:\\Users\\张三\\Desktop\\test.txt"

# 创建并启动小说阅读器
novel_reader = NovelReader(file_path)
