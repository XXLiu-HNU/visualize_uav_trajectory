---
name: visualize-uav-trajectory
description: Create UAV/drone video trajectory overlays, chronophotography, 渐变残影 and 实物实验轨迹叠影 from real footage, with paper figure layouts and editable PPTX output. Use for transport, grasping, placement and release illustrations, not metric ROS bag/CSV plots or fictional flight imagery.
---

# UAV video chronophotography

把真实视频中的多个时刻叠到一张实景图里。保留源像素、时间和蒙版，便于核对与复现；根据每段视频重新选帧，不沿用其他实验的坐标。

## 工作流

1. **检查输入。** 读取参考图及视频元数据，导出带时间的联系表并实际查看。先明确用户要全流程、运输至释放或末端局部；已有选择直接沿用。仅在范围会改变出图时提一个问题，期间继续检查素材。
2. **选择方法。** 固定镜头、静止背景的快速预览可用原仓库 `legacy`。摆动窗帘、人物、阴影、慢速悬停、需要清晰末端帧时，用本 skill 的关键帧局部蒙版。原仓库 `improved` 有颜色、左侧起始及向右单调运动假设，不能当成任意方向的真实跟踪；它会缓存片段全部帧，长4K片段不要直接套用。镜头移动明显时，先配准并核查静态地标，或选稳定片段；本脚本不自动稳像。
3. **选真实帧。** 粗略扫片后细看动作附近。选择能看清释放/放置的背景帧，并按空间间距选择较早机影，减少悬停重叠。把箱体、夹爪、分离载荷一起考虑。记录请求时间和实际解码帧编号；背景帧不等于精确释放时刻。
4. **保存配置并合成。** 使用 [脚本与配置说明](references/rendering.md)。原图像素选框，框内归一化多边形。自动分割失败时先检查原裁剪：漏载荷用补充蒙版，带入背景用排除蒙版，复杂帧用人工轮廓覆盖。保留分离部件，不只留下最大连通域。在最终排版尺寸下确认最早机影仍可辨认，必要时提高不透明度下限，不机械套用脚本默认值；轮廓属于人工辅助，不宣称全自动识别。
5. **逐帧审核。** 打开 `mask_audit.jpg`、`terminal.png` 和最终图，核对螺旋桨、夹爪、载荷、窗帘/网格/障碍物误入及机影间距。验证尺寸、背景区域一致性和真实帧来源。若局部反复失败，回看原裁剪并直接修正该帧轮廓或更换关键帧，避免反复全片调阈值。
6. **排版与交付。** 阅读 [最终排版与可编辑 PPTX](references/publication.md)。默认交付原分辨率无标注 PNG、排版图、小预览、可编辑 `.pptx`、配置、manifest及复现命令；用户明确限定格式时按其要求缩小输出范围。论文图按目标版式增加 PDF/SVG/TIFF 与英文图注。PPT 中标题、主图和末端放大必须是独立可编辑对象；不能只把整张合图贴进幻灯片就称为可编辑版。局部放大从真实源帧裁剪，保留无标注原图并披露编辑范围和残余抠图问题。

## 命令入口

`<skill-dir>` 是当前 `SKILL.md` 所在目录，无需原仓库在场。使用能导入 `cv2` 和 `numpy` 的 Python；缺依赖时在项目虚拟环境安装 `opencv-python-headless numpy`。

```bash
python "<skill-dir>/scripts/video_chronophoto.py" inspect \
  --video flight.mp4 --times 0,10,20 --output output/inspection
python "<skill-dir>/scripts/video_chronophoto.py" render \
  --video flight.mp4 --config keyframes.json --output output/render
```

先用 `--help` 核对选项。联系表仅用于定位；绘制蒙版时查看对应原帧，不把缩略图坐标当作原图坐标。
此脚本负责叠影与审核材料，不直接生成 PPTX。最终排版和 PPTX 由助手使用当前环境可用的图片、演示文稿工具完成；不得把 PNG 改扩展名作为 PPTX。

## 证据边界

- 使用原始画面进行可解释的合成；不生成、移动或补画无人机、载荷、飞行路径来冒充观测结果。
- 非等间隔机影不能直接比较速度。米制轨迹、速度和命中精度需要额外标定/日志；仅凭文件名不推断任务成功。
- 示例尺寸、帧率、时间、任务名称和取框均不是默认实验参数。多个视频分别出配置。
- 对于可变帧率视频，脚本的秒数换算是名义 FPS 近似；需要精确事件时序时另查原始 PTS，不把近似值写成测量真值。
