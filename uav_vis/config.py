import os

DEFAULT_DIR = os.path.join(os.getcwd(), "images")
DEFAULT_FILE = "default.png"
DISPLAY_HEIGHT_RATIO = 1.8

DEFAULT_PARAMS = {
    "sample_interval": 10,
    "diff_threshold": 30,
    "kernel_size": 15,
    "start_time": 0.0,
    "end_time": 10.0,
    "alpha_start": 0.2,
    "alpha_end": 1.0,
    "line_style": "white_orange",
    "line_thickness_scale": 1.0,
    "glow_strength": 0.3,
}

MAX_BACKGROUND_FRAMES = 30
MIN_TRACK_POINTS = 8
EXPECTED_TRACK_Y_RATIO = 0.49

MODE_OPTIONS = {
    "legacy": "原始残影叠加",
    "improved": "改进混合模式",
}

MODE_HINTS = {
    "legacy": "更稳定，适合快速得到残影轨迹图。",
    "improved": "会在残影上叠加平滑轨迹线，效果更适合展示，但对视频质量更敏感。",
}

IMPROVED_LINE_STYLES = {
    "white_orange": {
        "label": "亮白主线 + 橙色光晕",
        "main": (245, 245, 245),
        "glow": (0, 150, 255),
        "preview": "#f5f5f5",
    },
    "bright_yellow": {
        "label": "亮黄色",
        "main": (0, 255, 255),
        "glow": (0, 210, 255),
        "preview": "#ffd400",
    },
    "bright_pink": {
        "label": "亮粉色",
        "main": (255, 160, 255),
        "glow": (255, 60, 255),
        "preview": "#ff7ef3",
    },
    "bright_cyan": {
        "label": "亮青色",
        "main": (255, 255, 0),
        "glow": (255, 180, 0),
        "preview": "#49f2ff",
    },
}

