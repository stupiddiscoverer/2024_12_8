import os
import random
import shutil
import subprocess
import threading
import time

from pynput.keyboard import Key, Listener


class KeyTime:
    def __init__(self, key='', press_time=0.0, start=True):
        self.key = key
        self.press_time = press_time
        self.start = start

    def set(self, key='', press_time=0.0, start=True):
        self.key = key
        self.press_time = press_time
        self.start = start


class KeyQueue:
    def __init__(self, length=100):
        self.head = 0
        self.end = 0
        self.length = length
        self.list = [KeyTime()] * length
        self.lock = threading.Lock()

    def put(self, key='', press_time=0.0, start=True):
        with self.lock:
            end = (self.end+1) % self.length
            if end == self.head:
                return
            pre = (self.end-1) % self.length
            if self.list[pre].key == key and self.list[pre].start == start:
                return
            self.list[self.end].set(key, press_time, start)
            self.end = end

    def get(self):
        with self.lock:
            if self.head == self.end:
                return None
            key_time = self.list[self.head]
            self.head = (self.head + 1) % self.length
            return key_time


def key_to_str(key):
    if hasattr(key, 'char'):
        key = key.char
        if 1 <= ord(key) <= 26:
            return chr(ord(key) + 96)
        return key
    else:
        return str(key)


class KeyPressMonitor:
    def __init__(self):
        self.key_queue = KeyQueue()

    def on_press(self, key):
        key = key_to_str(key)
        print('pressing ...', key)
        self.key_queue.put(key, time.time(), True)

    def on_release(self, key):
        key = key_to_str(key)
        print('releasing ...', key)
        self.key_queue.put(key, time.time(), False)

    def start_listener(self):
        with Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()


class Node:
    def __init__(self, music):
        self.music = music
        self.prev = None
        self.next = None


class DoublyLinkedList:
    def __init__(self, musics):
        self.head = None
        self.tail = None
        self.current = None
        self.create_list(musics)

    def create_list(self, musics):
        if not musics:
            return
        self.head = Node(musics[0])
        self.current = self.head
        last = self.head
        for music in musics[1:]:
            new_node = Node(music)
            last.next = new_node
            new_node.prev = last
            last = new_node
        self.tail = last

    def delete_current(self):
        if not self.current:
            return
        abs_path = os.path.join(path, self.current.music)
        print(f"deleting: {abs_path}")  # 调试信息
        next_node = self.current.next
        prev_node = self.current.prev

        if self.current == self.head:
            self.head = next_node
        if self.current == self.tail:
            self.tail = prev_node
        if prev_node:
            prev_node.next = next_node
        if next_node:
            next_node.prev = prev_node

        self.current = next_node
        os.remove(abs_path)
        if os.path.exists(abs_path[0:-4] + '.lrc'):
            os.remove(abs_path[0:-4] + '.lrc')

    def like_current(self):
        if not self.current or self.current.music.startswith('like\\'):
            return
        like_folder = 'like'
        base_name = os.path.basename(self.current.music)
        new_path = os.path.join(path, like_folder)
        os.makedirs(new_path, exist_ok=True)
        print('moving', os.path.join(path, self.current.music), os.path.join(new_path, base_name))
        shutil.move(os.path.join(path, self.current.music),
                    os.path.join(new_path, base_name))
        if os.path.exists(os.path.join(path, self.current.music[0:-4]+'.lrc')):
            shutil.move(os.path.join(path, self.current.music[0:-4]+'.lrc'),
                        os.path.join(new_path, base_name[0:-4]+'.lrc'))
        self.current.music = os.path.join(like_folder, base_name)

    def next_music(self):
        if not self.current:
            return
        if not self.current.music.__contains__('\\'):
            watch_folder = 'watched'
            new_path = os.path.join(path, watch_folder)
            os.makedirs(new_path, exist_ok=True)
            print('moving', os.path.join(path, self.current.music), os.path.join(new_path, self.current.music))
            shutil.move(os.path.join(path, self.current.music),
                        os.path.join(new_path, self.current.music))
            if os.path.exists(os.path.join(path, self.current.music[0:-4]+'.lrc')):
                shutil.move(os.path.join(path, self.current.music[0:-4]+'.lrc'),
                            os.path.join(new_path, self.current.music[0:-4]+'.lrc'))
            self.current.music = os.path.join(watch_folder, self.current.music)
        self.current = self.current.next

    def prev_music(self):
        if self.current and self.current.prev:
            self.current = self.current.prev


def get_music_files(path='.'):
    extensions = (".mp3", ".ogg")
    return [f for f in os.listdir(path) if f.endswith(extensions)]


def exec_order(process, order=''):
    process.stdin.write(order.encode())
    process.stdin.flush()


def play_music(playlist, path='.'):
    while playlist.current:
        music_path = os.path.join(path, playlist.current.music)
        print(f"Playing: {music_path}")  # 调试信息
        process = subprocess.Popen(["ffplay", "-autoexit", music_path],
                                   stdin=subprocess.PIPE,  stdout=subprocess.DEVNULL)
        ctrl = False
        next_music = True
        try:
            while process.poll() is None:
                next_music = True
                key_time = monitor.key_queue.get()
                if key_time is None:
                    continue
                if key_time.key.__contains__(str(Key.esc)):
                    process.terminate()
                    return
                if key_time.key.__contains__('ctrl'):
                    if key_time.start:
                        print('ctrl start...')
                        ctrl = True
                    else:
                        print('ctrl end...')
                        ctrl = False
                    continue
                if ctrl:
                    print(len(key_time.key))
                    if key_to_str(Key.delete) == key_time.key:  # 删除当前视频
                        print('delete')
                        process.terminate()
                        process.wait(0.1)
                        playlist.delete_current()
                        next_music = False
                        break
                    if key_to_str(Key.insert) == key_time.key:
                        print('like')
                        process.terminate()
                        process.wait(0.1)
                        playlist.like_current()
                        break
                if 'n' == key_time.key:
                    print('next')
                    break
                if 'b' == key_time.key:
                    print('back')
                    next_music = False
                    process.terminate()
                    playlist.prev_music()
                    break
            print('...........')
            if next_music:
                process.terminate()
                process.wait(0.1)
                playlist.next_music()
        except Exception as e:
            print(f"发生错误：{e}")
            quit()


if __name__ == '__main__':
    path = os.path.abspath('.')
    music_files = get_music_files(path)
    if len(music_files) < 1:
        path = 'C:\\Users\\张三\\Music\\7.怀旧情怀金典200首'
        music_files = get_music_files(path)
    random.shuffle(music_files)
    playlist = DoublyLinkedList(music_files)

    # 创建监听器实例
    monitor = KeyPressMonitor()
    # 在后台启动监听器（在另一个线程中）
    listener_thread = threading.Thread(target=monitor.start_listener)
    listener_thread.daemon = True  # 设置为守护线程，这样主程序退出时线程也会退出
    listener_thread.start()

    if playlist.head:
        play_music(playlist, path)
    else:
        print("No musics available to play.")
