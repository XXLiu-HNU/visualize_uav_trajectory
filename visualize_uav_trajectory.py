import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import Scale, HORIZONTAL, Button, Spinbox
from PIL import Image, ImageTk
from tkinter import filedialog

# 常量定义
DEFAULT_DIR = os.path.join(os.getcwd(), "images") # 默认保存路径
DEFAULT_FILE = "default.png" # 默认保存文件名
DISPLAY_HEIGHT_RATIO = 1.8 # 默认显示比例

def overlay_drone_trajectory(video_path, sample_interval, diff_threshold, kernel_size):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}！请检查路径。")
        return None

    ret, first_frame = cap.read()
    if not ret:
        print("无法读取视频的第一帧！")
        return None

    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    background_image = first_frame.copy()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % sample_interval == 0:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_diff = cv2.absdiff(gray_frame, first_gray)
            _, diff_mask = cv2.threshold(frame_diff, diff_threshold, 255, cv2.THRESH_BINARY)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            dilated_mask = cv2.dilate(diff_mask, kernel, iterations=1)
            motion_only = cv2.bitwise_and(frame, frame, mask=dilated_mask)
            background_image[dilated_mask == 255] = motion_only[dilated_mask == 255]

        frame_count += 1

    cap.release()
    return background_image

def update_image():
    sample_interval = sample_interval_scale.get()
    diff_threshold = diff_threshold_scale.get()
    kernel_size = kernel_size_scale.get()
    result_image = overlay_drone_trajectory(video_path, sample_interval, diff_threshold, kernel_size)
    if result_image is not None:
        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(result_image_rgb)

        # 输出原图大小
        original_width, original_height = img.size
        print(f"Original Image width: {original_width}")
        print(f"Original Image height: {original_height}")

        # 调整图像大小，保持比例
        display_height = int(default_height / DISPLAY_HEIGHT_RATIO)
        aspect_ratio = img.width / img.height
        display_width = int(display_height * aspect_ratio)
        display_img = img.resize((display_width, display_height), Image.LANCZOS)

        print(f"Display Image width: {display_width}")
        print(f"Display Image height: {display_height}")

        imgtk = ImageTk.PhotoImage(image=display_img)
        result_label.config(image=imgtk)
        result_label.image = imgtk

        # 将图像保存到全局变量中，以便稍后保存
        global saved_image
        saved_image = result_image

        # 使用 pack 方法设置图片的位置
        result_label.pack(side="top", pady=(100, 100))

def save_image():
    if saved_image is not None:
        if not os.path.exists(DEFAULT_DIR):
            os.makedirs(DEFAULT_DIR)

        # 打开文件保存对话框
        file_path = filedialog.asksaveasfilename(initialdir=DEFAULT_DIR,
                                                initialfile=DEFAULT_FILE,
                                                defaultextension=".png",
                                                filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if file_path:
            # 保存图像
            cv2.imwrite(file_path, saved_image)
            print(f"图像已保存到: {file_path}")

video_path = "test_video.mp4"
saved_image = None # 用于保存当前图像

root = tk.Tk()
root.title("无人机轨迹参数调整")

# 设置背景颜色
# root.configure(bg="")

# 获取显示器分辨率
screen_width = int(root.winfo_screenwidth())
screen_width = int(screen_width / 2) # 如果你使用的是双屏显示器，那么取消这一行的注释
screen_height = int(root.winfo_screenheight())
print(f"Screen width: {screen_width}")
print(f"Screen height: {screen_height}")

# 设置GUI界面的默认大小
default_width = int(screen_width * 0.5)
default_height = int(screen_height * 0.7)
root.geometry(f"{default_width}x{default_height}")

# 通过设置采样间隔，可以控制处理视频的频率。
# 较大的采样间隔会减少处理的帧数，从而提高处理速度，但可能会遗漏一些细节；
# 较小的采样间隔会增加处理的帧数，从而提高精度，但会增加处理时间。
sample_interval_scale = Scale(root, from_=1, to_=100, orient=HORIZONTAL, label="采样间隔")
sample_interval_scale.set(10)
sample_interval_scale.pack()

# 通过设置差分阈值，可以控制检测到的运动区域。
# 较低的阈值会检测到更多的运动，但可能会包含噪声；
# 较高的阈值会减少噪声，但可能会遗漏一些细微的运动。
diff_threshold_scale = Scale(root, from_=1, to_=255, orient=HORIZONTAL, label="差分阈值")
diff_threshold_scale.set(30)
diff_threshold_scale.pack()

# 通过设置膨胀核大小，可以控制膨胀操作的强度。
# 较大的核会使运动区域变得更大、更连贯，但可能会导致过度膨胀；
# 较小的核会使运动区域保持较小的尺寸，但可能会导致不连贯的结果。
kernel_size_scale = Scale(root, from_=1, to_=50, orient=HORIZONTAL, label="膨胀核大小")
kernel_size_scale.set(15)
kernel_size_scale.pack()

update_button = Button(root, text="更新图像", command=update_image)
update_button.pack()

save_button = Button(root, text="保存图片", command=save_image)
save_button.pack()

result_label = tk.Label(root)
result_label.pack()

root.mainloop()