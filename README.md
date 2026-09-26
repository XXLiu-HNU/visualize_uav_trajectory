# A tool to visualize the trajectory of drones in videos

## 🧑🏽‍🦽🧑🏽‍🦽🧑🏽‍🦽快速开始 💥💥💥

得益于AGENT的日益强大， 目前已经不需要复杂的调参过程了，直接让codex阅读本项目中的代码、封装的skills，并且把你想要复现的图片加上即可一句话实现同款顶刊配图。

**使用 Skill**：把飞行视频和参考图交给 Codex，生成实景轨迹叠影、末端局部放大和可编辑 PPT，保留关键帧配置与复现命令。选帧和蒙版由助手处理并核查，也可以继续用自然语言调整。

### 1. 在本仓库中直接调用

在 Codex 中打开本仓库，输入下面的提示词，将路径换成你的素材：

```text
使用 $visualize-uav-trajectory，处理 /data/flight.mp4。
参考 /data/ref.png，只展示运输至释放阶段。
输出原分辨率叠影、末端局部放大、论文合图和可编辑 PPTX，
保留关键帧配置与复现命令。
```

没有参考图时省略参考路径，说明想展示的动作即可。处理多个实验时，直接提供多个视频路径。
如果用于论文，补充期刊或栏宽要求，例如“IEEE RA-L，双栏通栏图”。

仓库包含 `.agents/skills/visualize-uav-trajectory` 发现入口。如果 Codex 尚未发现它，
可重新启动 Codex，或直接让助手读取 [SKILL.md](skills/visualize-uav-trajectory/SKILL.md)。

### 2. 在其他项目中安装后调用

先向 Codex 输入一次：

```text
使用 $skill-installer，从 XXLiu-HNU/visualize_uav_trajectory
安装 skills/visualize-uav-trajectory。
```

安装后，使用上面的 `$visualize-uav-trajectory` 提示词即可。Skill 的渲染脚本独立于原有 GUI；
最终排版和 PPTX 导出由助手使用当前环境中的相应工具完成。

### 3. 用自然语言继续调整

```text
保留现有选帧，把最初几个无人机叠影调得更清楚。
两张主图保留完整宽度，统一亮度和色调，把标题放进照片里。
局部放大不要遮挡轨迹，并同步更新可编辑 PPTX。
```

不需要先学习参数表。助手会结合实际画面修改配置，并保留原图供核对。

### 最终会得到什么

| 输出 | 用途 |
|---|---|
| 原分辨率叠影 PNG、末端原帧 | 查看动作并核对真实画面 |
| 排版合图、小预览 | 论文插图或结果展示；按需求补充 PDF、SVG、TIFF |
| 可编辑 PowerPoint（`.pptx`） | 分别调整文字、主图与局部放大图的位置、尺寸和裁切 |
| 关键帧配置、蒙版审核、manifest、复现命令 | 核查来源并重新出图 |

PPT 中的文字是文本框，主图和局部图是独立图片对象。历史无人机叠影默认已经合成在主图中；
要修改单个机影的透明度，继续让 Skill 调整配置后重新生成。
真实照片的清晰度受原视频限制，叠影展示也不代替米制轨迹或速度测量。

## 原有示例

以下保留项目原有 GIF 和效果图，用于了解不同素材的叠影效果。

### 视频与叠影结果

![gif](./example/speed2-1.gif)

![result](./example/4tree.png)

### 更多效果

![result](./example/update.jpg)

视频来源：[composite_image](https://github.com/RENyunfan/composite_image)。

![result](./example/update2.png)

### Improved 模式与对照

![improved](./example/example_improved.png)

![comparison](./example/example_comparison.png)

### 原有 GUI

![new_gui](./example/new_gui.png)

## 进一步阅读

- [手动操作与旧版工具使用手册](docs/MANUAL_USAGE.md)：原有 CLI、GUI、参数说明、模式差异与项目结构。
- [Skill 完整流程](skills/visualize-uav-trajectory/SKILL.md)：视频检查、蒙版审核、最终排版与交付要求。

欢迎提出改进建议，也欢迎给项目点个 Star ⭐。
