import os
import tkinter as tk
from tkinter import Button, Entry, Label, OptionMenu, Scale, filedialog

import cv2
from PIL import Image, ImageTk

from .config import DEFAULT_DIR, DEFAULT_FILE, DEFAULT_PARAMS, DISPLAY_HEIGHT_RATIO, IMPROVED_LINE_STYLES, MODE_HINTS, MODE_OPTIONS
from .core import RenderParams, overlay_drone_trajectory


class TrajectoryGUI:
    def __init__(self):
        self.video_path = ""
        self.saved_image = None
        self.root = tk.Tk()
        self.root.title("无人机轨迹参数调整")
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.default_width = int(screen_width * 0.5)
        self.default_height = int(screen_height * 0.78)
        self.root.geometry(f"{self.default_width}x{self.default_height}")

        self.method_var = tk.StringVar(value="legacy")
        self.method_display_var = tk.StringVar(value=MODE_OPTIONS["legacy"])
        self.line_style_var = tk.StringVar(value=DEFAULT_PARAMS["line_style"])
        self.line_style_display_var = tk.StringVar(value=IMPROVED_LINE_STYLES[DEFAULT_PARAMS["line_style"]]["label"])

        self._build_ui()
        self._sync_mode_display()
        self._sync_line_style_display()
        self.preview_frame.bind("<Configure>", self.on_preview_resize)

    def _build_ui(self):
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill="both", expand=True)

        self.controls_frame = tk.Frame(self.main_frame, width=320)
        self.controls_frame.pack(side="left", fill="y", padx=10, pady=10)
        self.controls_frame.pack_propagate(False)

        self.preview_frame = tk.Frame(self.main_frame)
        self.preview_frame.pack(side="right", fill="both", expand=True, padx=(0, 10), pady=10)

        self.file_label = Label(self.controls_frame, text="未选择文件", wraplength=300)
        self.file_label.pack()
        Button(self.controls_frame, text="选择视频文件", command=self.select_file).pack(fill="x")

        Label(self.controls_frame, text="模式").pack()
        OptionMenu(self.controls_frame, self.method_display_var, *MODE_OPTIONS.values(), command=self.on_method_change).pack(fill="x")
        self.mode_hint_label = Label(self.controls_frame, text="", wraplength=300, justify="left")
        self.mode_hint_label.pack()

        self.line_style_label = Label(self.controls_frame, text="改进线条颜色")
        self.line_style_label.pack()
        self.line_style_menu = OptionMenu(
            self.controls_frame,
            self.line_style_display_var,
            *[style["label"] for style in IMPROVED_LINE_STYLES.values()],
            command=self.on_line_style_change,
        )
        self.line_style_menu.pack(fill="x")
        self.color_preview = Label(self.root, text="      ", bg=IMPROVED_LINE_STYLES[DEFAULT_PARAMS["line_style"]]["preview"])
        self.color_preview.pack(in_=self.controls_frame)

        Button(self.controls_frame, text="推荐参数", command=self.apply_recommended_params).pack(fill="x")

        self.sample_interval_entry = self._add_entry("采样间隔", DEFAULT_PARAMS["sample_interval"])
        self.diff_threshold_entry = self._add_entry("差分阈值", DEFAULT_PARAMS["diff_threshold"])
        self.kernel_size_entry = self._add_entry("膨胀核大小", DEFAULT_PARAMS["kernel_size"])
        self.start_time_entry = self._add_entry("视频起始时间（秒）", DEFAULT_PARAMS["start_time"])
        self.end_time_entry = self._add_entry("视频结束时间（秒）", DEFAULT_PARAMS["end_time"])
        self.alpha_start_entry = self._add_entry("透明度起始值", DEFAULT_PARAMS["alpha_start"])
        self.alpha_end_entry = self._add_entry("透明度最终值", DEFAULT_PARAMS["alpha_end"])

        self.line_thickness_label = Label(self.controls_frame, text="轨迹线条粗细")
        self.line_thickness_label.pack()
        self.line_thickness_scale = Scale(self.controls_frame, from_=0.5, to=2.5, resolution=0.1, orient=tk.HORIZONTAL)
        self.line_thickness_scale.set(DEFAULT_PARAMS["line_thickness_scale"])
        self.line_thickness_scale.pack(fill="x")

        self.glow_strength_label = Label(self.controls_frame, text="轨迹光晕强度")
        self.glow_strength_label.pack()
        self.glow_strength_scale = Scale(self.controls_frame, from_=0.0, to=0.8, resolution=0.05, orient=tk.HORIZONTAL)
        self.glow_strength_scale.set(DEFAULT_PARAMS["glow_strength"])
        self.glow_strength_scale.pack(fill="x")

        Button(self.controls_frame, text="更新图像", command=self.update_image).pack(fill="x")
        Button(self.controls_frame, text="保存图片", command=self.save_image).pack(fill="x")

        self.status_label = Label(self.controls_frame, text="请选择视频并设置参数。", wraplength=300, justify="left")
        self.status_label.pack(fill="x", pady=(8, 0))
        self.result_label = Label(self.preview_frame, anchor="center")
        self.result_label.pack(fill="both", expand=True)

    def _add_entry(self, label, value):
        Label(self.controls_frame, text=label).pack()
        entry = Entry(self.controls_frame)
        entry.pack(fill="x")
        entry.insert(0, str(value))
        return entry

    def _sync_mode_display(self):
        self.method_var.set(next(key for key, value in MODE_OPTIONS.items() if value == self.method_display_var.get()))
        self.mode_hint_label.config(text=MODE_HINTS[self.method_var.get()])
        is_improved = self.method_var.get() == "improved"
        state = tk.NORMAL if is_improved else tk.DISABLED
        self.line_style_menu.config(state=state)
        self.line_thickness_scale.config(state=state)
        self.glow_strength_scale.config(state=state)
        label_color = "black" if is_improved else "gray45"
        self.line_style_label.config(fg=label_color)
        self.line_thickness_label.config(fg=label_color)
        self.glow_strength_label.config(fg=label_color)
        self.color_preview.config(bg=IMPROVED_LINE_STYLES[self.line_style_var.get()]["preview"] if is_improved else "#d9d9d9")

    def _sync_line_style_display(self):
        current_label = self.line_style_display_var.get()
        for key, style in IMPROVED_LINE_STYLES.items():
            if style["label"] == current_label:
                self.line_style_var.set(key)
                self.color_preview.config(bg=style["preview"])
                return

    def on_method_change(self, _selection):
        self._sync_mode_display()
        self.set_status("已切换模式。")

    def on_line_style_change(self, _selection):
        self._sync_line_style_display()
        self.set_status("已更新改进线条颜色预览。")

    def apply_recommended_params(self):
        presets = {
            "legacy": {
                "sample_interval": 4,
                "diff_threshold": 28,
                "kernel_size": 11,
                "alpha_start": 0.2,
                "alpha_end": 1.0,
            },
            "improved": {
                "sample_interval": 2,
                "diff_threshold": 28,
                "kernel_size": 11,
                "alpha_start": 0.2,
                "alpha_end": 1.0,
            },
        }
        preset = presets[self.method_var.get()]
        self._set_entry(self.sample_interval_entry, preset["sample_interval"])
        self._set_entry(self.diff_threshold_entry, preset["diff_threshold"])
        self._set_entry(self.kernel_size_entry, preset["kernel_size"])
        self._set_entry(self.alpha_start_entry, preset["alpha_start"])
        self._set_entry(self.alpha_end_entry, preset["alpha_end"])
        if self.method_var.get() == "improved":
            self.line_thickness_scale.set(1.2)
            self.glow_strength_scale.set(0.25)
        else:
            self.line_thickness_scale.set(1.0)
            self.glow_strength_scale.set(0.3)
        self.set_status("已应用推荐参数。")

    @staticmethod
    def _set_entry(entry, value):
        entry.delete(0, tk.END)
        entry.insert(0, str(value))

    def parse_ui_values(self):
        try:
            return RenderParams(
                sample_interval=int(self.sample_interval_entry.get()),
                diff_threshold=int(self.diff_threshold_entry.get()),
                kernel_size=int(self.kernel_size_entry.get()),
                start_time=float(self.start_time_entry.get()),
                end_time=float(self.end_time_entry.get()),
                alpha_start=float(self.alpha_start_entry.get()),
                alpha_end=float(self.alpha_end_entry.get()),
                line_style=self.line_style_var.get(),
                line_thickness_scale=float(self.line_thickness_scale.get()),
                glow_strength=float(self.glow_strength_scale.get()),
            )
        except ValueError:
            self.set_status("请输入有效的数值。", is_error=True)
            return None

    def render_image_for_display(self, result_image):
        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(result_image_rgb)
        self.root.update_idletasks()
        available_width = max(200, self.preview_frame.winfo_width() - 20)
        available_height = max(200, self.preview_frame.winfo_height() - 20)
        width_ratio = available_width / img.width
        height_ratio = available_height / img.height
        scale_ratio = min(width_ratio, height_ratio, 1.0)
        display_width = max(1, int(img.width * scale_ratio))
        display_height = max(1, int(img.height * scale_ratio))
        display_img = img.resize((display_width, display_height), Image.LANCZOS)
        return ImageTk.PhotoImage(image=display_img)

    def update_image(self):
        params = self.parse_ui_values()
        if params is None:
            return
        if not self.video_path:
            self.set_status("请先选择视频文件。", is_error=True)
            return
        self.set_status("处理中...")
        result_image = overlay_drone_trajectory(self.video_path, params, method=self.method_var.get(), notifier=self.show_gui_message)
        if result_image is None:
            return
        imgtk = self.render_image_for_display(result_image)
        self.result_label.config(image=imgtk)
        self.result_label.image = imgtk
        self.saved_image = result_image
        self.set_status("处理完成。")

    def on_preview_resize(self, _event):
        if self.saved_image is None:
            return
        imgtk = self.render_image_for_display(self.saved_image)
        self.result_label.config(image=imgtk)
        self.result_label.image = imgtk

    def save_image(self):
        if self.saved_image is None:
            self.set_status("当前没有可保存的图像。", is_error=True)
            return
        if not os.path.exists(DEFAULT_DIR):
            os.makedirs(DEFAULT_DIR)
        file_path = filedialog.asksaveasfilename(
            initialdir=DEFAULT_DIR,
            initialfile=DEFAULT_FILE,
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
        )
        if file_path:
            cv2.imwrite(file_path, self.saved_image)
            self.set_status(f"图像已保存到: {file_path}")

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
        if file_path:
            self.video_path = file_path
            self.file_label.config(text=f"已选择文件: {file_path}")
            self.set_status("视频已选择。")

    def set_status(self, message, is_error=False):
        self.status_label.config(text=message, fg="crimson" if is_error else "black")

    def show_gui_message(self, message):
        self.set_status(message, is_error=True)

    def run(self):
        self.root.mainloop()


def run_gui():
    TrajectoryGUI().run()
