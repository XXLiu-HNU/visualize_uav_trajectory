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

def overlay_drone_trajectory(video_path, sample_interval, diff_threshold, kernel_size, start_time, end_time, alpha_start, alpha_end):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}！请检查路径。")
        return None

    # 获取视频的总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    # 跳转到指定的起始时间
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)  # 起始时间（秒转毫秒）

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

        # 如果视频已经到达结束时间，停止
        if cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 >= end_time:
            break

        if frame_count % sample_interval == 0:
            # 转换当前帧为灰度图
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 计算当前帧与第一帧的差异
            frame_diff = cv2.absdiff(gray_frame, first_gray)
            _, diff_mask = cv2.threshold(frame_diff, diff_threshold, 255, cv2.THRESH_BINARY)
            
            # 形态学膨胀操作
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            dilated_mask = cv2.dilate(diff_mask, kernel, iterations=1)
            
            # 提取运动区域
            motion_only = cv2.bitwise_and(frame, frame, mask=dilated_mask)

            # 计算透明度：从 alpha_start 到 alpha_end
            alpha = alpha_start + (alpha_end - alpha_start) * (frame_count / total_frames)  # 透明度逐渐变化
            beta = 1 - alpha  # 背景图的透明度

            # 使用加权平均法进行透明叠加
            # 只在掩膜为255的地方进行加权叠加
            for c in range(3):  # 3通道的图像，逐通道处理
                background_image[:, :, c] = (background_image[:, :, c] * beta + motion_only[:, :, c] * alpha) * (dilated_mask == 255) + background_image[:, :, c] * (dilated_mask != 255)

        frame_count += 1

    cap.release()

    # 关闭所有图像窗口
    cv2.destroyAllWindows()

    return background_image

def update_image():
    try:
        sample_interval = int(sample_interval_entry.get())
        diff_threshold = int(diff_threshold_entry.get())
        kernel_size = int(kernel_size_entry.get())
        start_time = float(start_time_entry.get())
        end_time = float(end_time_entry.get())
        alpha_start = float(alpha_start_entry.get())
        alpha_end = float(alpha_end_entry.get())
    except ValueError:
        print("请输入有效的数值！")
        return

    result_image = overlay_drone_trajectory(video_path, sample_interval, diff_threshold, kernel_size, start_time, end_time, alpha_start, alpha_end)
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

Label(root, text="视频起始时间（秒）").pack()
start_time_entry = Entry(root)
start_time_entry.pack()
start_time_entry.insert(0, "0")

Label(root, text="视频结束时间（秒）").pack()
end_time_entry = Entry(root)
end_time_entry.pack()
end_time_entry.insert(0, "10")

Label(root, text="透明度起始值").pack()
alpha_start_entry = Entry(root)
alpha_start_entry.pack()
alpha_start_entry.insert(0, "0.2")

Label(root, text="透明度最终值").pack()
alpha_end_entry = Entry(root)
alpha_end_entry.pack()
alpha_end_entry.insert(0, "1")

update_button = Button(root, text="更新图像", command=update_image)
update_button.pack()
save_button = Button(root, text="保存图片", command=save_image)
save_button.pack()

result_label = Label(root)
result_label.pack()

root.mainloop()
