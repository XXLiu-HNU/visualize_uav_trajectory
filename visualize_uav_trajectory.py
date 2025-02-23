import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Entry, Button
from PIL import Image, ImageTk

# 常量定义
DEFAULT_DIR = os.path.join(os.getcwd(), "images")
DEFAULT_FILE = "default.png"
DISPLAY_HEIGHT_RATIO = 1.8

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
    try:
        sample_interval = int(sample_interval_entry.get())
        diff_threshold = int(diff_threshold_entry.get())
        kernel_size = int(kernel_size_entry.get())
    except ValueError:
        print("请输入有效的数值！")
        return

    result_image = overlay_drone_trajectory(video_path, sample_interval, diff_threshold, kernel_size)
    if result_image is not None:
        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(result_image_rgb)

        # 调整图像大小，保持比例
        display_height = int(default_height / DISPLAY_HEIGHT_RATIO)
        aspect_ratio = img.width / img.height
        display_width = int(display_height * aspect_ratio)
        display_img = img.resize((display_width, display_height), Image.LANCZOS)

        imgtk = ImageTk.PhotoImage(image=display_img)
        result_label.config(image=imgtk)
        result_label.image = imgtk

        global saved_image
        saved_image = result_image

def save_image():
    if saved_image is not None:
        if not os.path.exists(DEFAULT_DIR):
            os.makedirs(DEFAULT_DIR)
        file_path = filedialog.asksaveasfilename(initialdir=DEFAULT_DIR,
                                                 initialfile=DEFAULT_FILE,
                                                 defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if file_path:
            cv2.imwrite(file_path, saved_image)
            print(f"图像已保存到: {file_path}")

def select_file():
    global video_path
    file_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
    if file_path:
        video_path = file_path
        file_label.config(text=f"已选择文件: {file_path}")

video_path = ""
saved_image = None

root = tk.Tk()
root.title("无人机轨迹参数调整")
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
default_width = int(screen_width * 0.5)
default_height = int(screen_height * 0.7)
root.geometry(f"{default_width}x{default_height}")

file_label = Label(root, text="未选择文件", wraplength=default_width)
file_label.pack()
file_button = Button(root, text="选择视频文件", command=select_file)
file_button.pack()

Label(root, text="采样间隔").pack()
sample_interval_entry = Entry(root)
sample_interval_entry.pack()
sample_interval_entry.insert(0, "10")

Label(root, text="差分阈值").pack()
diff_threshold_entry = Entry(root)
diff_threshold_entry.pack()
diff_threshold_entry.insert(0, "30")

Label(root, text="膨胀核大小").pack()
kernel_size_entry = Entry(root)
kernel_size_entry.pack()
kernel_size_entry.insert(0, "15")

update_button = Button(root, text="更新图像", command=update_image)
update_button.pack()
save_button = Button(root, text="保存图片", command=save_image)
save_button.pack()

result_label = Label(root)
result_label.pack()

root.mainloop()
