import cv2
import numpy as np
from PIL import ImageGrab


def capture_screen(bbox=None):
    screen = np.array(ImageGrab.grab(bbox))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    return screen


def process_image(image):
    # 图像处理逻辑，如检测角色、敌人位置等
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 进行边缘检测等处理
    edges = cv2.Canny(gray_image, threshold1=200, threshold2=300)
    return edges


if __name__ == "__main__":
    while True:
        screen = capture_screen(bbox=(0, 0, 600, 600))  # 截取屏幕的一部分
        processed_screen = process_image(screen)
        cv2.imshow('Processed Screen', processed_screen)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
