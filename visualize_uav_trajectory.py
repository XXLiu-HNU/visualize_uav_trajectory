import cv2
import numpy as np
import tkinter as tk
from tkinter import Scale, HORIZONTAL, Button
from PIL import Image, ImageTk
from tkinter import filedialog

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
        imgtk = ImageTk.PhotoImage(image=img)
        result_label.config(image=imgtk)
        result_label.image = imgtk
        # 将图像保存到全局变量中，以便稍后保存
        global saved_image
        saved_image = result_image

def save_image():
    if saved_image is not None:
        # 打开文件保存对话框
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if file_path:
            # 保存图像
            cv2.imwrite(file_path, saved_image)
            print(f"图像已保存到: {file_path}")

video_path = "example_video.mp4"
saved_image = None  # 用于保存当前图像

root = tk.Tk()
root.title("无人机轨迹参数调整")

sample_interval_scale = Scale(root, from_=1, to_=100, orient=HORIZONTAL, label="采样间隔")
sample_interval_scale.set(10)
sample_interval_scale.pack()

diff_threshold_scale = Scale(root, from_=1, to_=255, orient=HORIZONTAL, label="差分阈值")
diff_threshold_scale.set(30)
diff_threshold_scale.pack()

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