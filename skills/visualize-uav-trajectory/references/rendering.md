# Portable renderer

脚本只依赖 Python 3、OpenCV 和 NumPy，不导入 GUI、`uav_vis` 或上层实验工作区。
视频须有可解码的 FPS/帧数元数据；按名义 FPS 就近取帧，帧号从 0 开始。
脚本保留原分辨率，每次只读取需要的帧。它提供图像坐标叠影，不提供目标跟踪或相机稳像。

## Inspect

```bash
python scripts/video_chronophoto.py inspect \
  --video /data/flight.mp4 --times 0,5,10 --output /data/result/inspection
```

输出 `metadata.json`、`contact_sheet.jpg` 及所选原始帧 PNG。按视频长度选择少量粗采样，
之后只在感兴趣的动作附近加密。不要全片提取4K PNG。输出目录使用专门的实验目录，避免覆盖素材。

## Render configuration

以下数值仅演示 schema，不适合直接套用其他视频：

```json
{
  "terminal_time": 4.0,
  "keyframes": [
    {
      "time": 1.0,
      "box": [80, 60, 240, 240],
      "opacity": 0.3
    },
    {
      "time": 2.5,
      "box": [300, 100, 480, 310],
      "opacity": 0.6,
      "manual_polygon": [[0.4, 0.05], [0.6, 0.05], [0.95, 0.4], [0.6, 0.95], [0.3, 0.95], [0.05, 0.4]],
      "include_polygons": [[[0.4, 0.85], [0.6, 0.85], [0.6, 0.98], [0.4, 0.98]]],
      "exclude_polygons": [[[0.8, 0], [1, 0], [1, 0.15], [0.8, 0.15]]]
    }
  ]
}
```

| 字段 | 语义 |
|---|---|
| `terminal_time` | 背景时刻（秒），必须在视频可解码范围内 |
| `keyframes` | 非空历史机影列表；时间严格递增，实际帧不能重复，且必须早于末端帧 |
| `time` | 历史源帧请求时间（秒） |
| `box` | 原图整数 `[x1,y1,x2,y2]`，右/下边界不含，必须在图内 |
| `opacity` | 可选0–1的不透明度，0为完全透明、1为完全不透明；省略时按源时间从0.25向0.72渐变 |
| `manual_polygon` | 可选人工完整轮廓；提供时绕过自动分割 |
| `include_polygons` | 可选多边形列表，用于补回载荷/机体漏检像素 |
| `exclude_polygons` | 可选多边形列表，用于去除背景；最后应用，因此排除优先 |

所有多边形坐标都是**当前选框内**的 `[u,v]`，范围0–1，至少3个点且面积非零。
如果在宽 `W`、高 `H` 的裁剪预览上点选 `(x,y)`，转换为 `u=x/(W-1), v=y/(H-1)`；
不要直接复用以前200×240审核缩略图的数值，也不要在新分辨率下复用旧像素框。
不提供 `manual_polygon` 时，GrabCut在框内工作；框必须包含主体并给四周留背景。

```bash
python scripts/video_chronophoto.py render \
  --video /data/flight.mp4 --config /data/keyframes.json --output /data/result/render
```

## Reviewable outputs

- `composite.png`：原分辨率合成图。
- `terminal.png`：对应末端真实帧，供背景与动作核查。
- `mask_audit.jpg`：每帧原裁剪与蒙版效果对照，检查是否完整保留载荷。
- 蒙版 PNG：逐帧可审核的最终叠影范围。
- `manifest.json`：输入视频名称、FPS、尺寸、请求/实际帧时刻、配置及验证结果。

文件成功写出不等于视觉验收。确认蒙版之外的背景与末端图一致，并对照源视频检查末端。
关键帧靠近末端时尤其注意历史蒙版不能覆盖清晰的末端飞机或载荷；出现重叠就换帧/缩框/排除区域。
在最终显示尺寸下检查最早机影。如果默认0.25太淡，可显式填写各帧 `opacity`，例如把起始值提高到0.55–0.65附近再视觉核对；这只是调节起点，不是所有视频的固定参数。
OpenCV无法提供可信时间戳时，manifest的实际时间是 `frame_index / fps`，不能作为可变帧率视频的PTS。

此渲染脚本只输出叠影证据和审核图；Skill 的完整交付还包含排版图与可编辑 PPTX，按
[最终排版与导出说明](publication.md)另行生成。标题、局部放大和多图对比保留为独立排版元素；
箭头若仅表达示意方向须明确标为示意，不填未经测量的数值。
