import cv2
import numpy as np

def overlay_drone_photo_on_background(video_path, output_image_path, sample_interval=5, diff_threshold=30):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}！请检查路径。")
        return

    ret, prev_frame = cap.read()
    if not ret:
        print("无法读取视频的第一帧！")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # 初始化背景图片
    background_image = prev_frame.copy()
    height, width, channels = prev_frame.shape

    frame_count = 0
    sampled_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % sample_interval == 0:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 计算帧差
            frame_diff = cv2.absdiff(gray_frame, prev_gray)

            # 二值化处理，提取变化区域
            _, diff_mask = cv2.threshold(frame_diff, diff_threshold, 255, cv2.THRESH_BINARY)

            # Debug: visualize the diff_mask (optional)
            # cv2.imwrite(f"debug_diff_mask_{frame_count}.jpg", diff_mask)

            # 提取运动区域的彩色部分
            motion_only = cv2.bitwise_and(frame, frame, mask=diff_mask)

            # Debug: visualize the motion_only before enhancement (optional)
            # cv2.imwrite(f"debug_motion_only_before_{frame_count}.jpg", motion_only)

            # 保留背景非运动部分
            static_background = cv2.bitwise_and(background_image, background_image, mask=cv2.bitwise_not(diff_mask))

            # Debug: visualize the static_background (optional)
            # cv2.imwrite(f"debug_static_background_{frame_count}.jpg", static_background)

            # 将运动区域覆盖到背景对应位置
            background_image = cv2.add(static_background, motion_only)

            # Debug: visualize the background image after update (optional)
            # cv2.imwrite(f"debug_background_image_{frame_count}.jpg", background_image)

            # 更新前一帧
            prev_gray = gray_frame
            sampled_frame_count += 1

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"已处理总帧数: {frame_count}，采样帧数: {sampled_frame_count}")

    # 保存最终图像
    cv2.imwrite(output_image_path, background_image)
    print(f"轨迹图片已保存到: {output_image_path}")

    cap.release()

# 调用函数
video_path = "test_video.mp4"  # 视频输入
output_image_path = "drone_trajectory_with_frame.jpg" # 图片输出
sample_interval = 5   # 采样率
diff_threshold = 100  # 差分阈值，越小对场景越敏感
# 保存图像
overlay_drone_photo_on_background(video_path, output_image_path, sample_interval, diff_threshold)