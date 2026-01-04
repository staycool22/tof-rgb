# tof-rgb

## usage

```bash
python tof_rgb.py /dev/tof /dev/rgb
# example:
# python tof_rgb.py /dev/video2 /dev/video4
```

## ToF 自动化测试套件 (Auto Test ToF)

`auto_test_tof_v2.py` 是一个用于 OPN600x ToF 模组的综合测试与评估工具。

### 功能特性
- **实时预览**: 深度图/红外图实时显示，支持 ROI 选择与数值监测。
- **数据解码**: 遵循官方协议，支持 16-bit Raw Code 解析与标定转换。
- **多项测试**:
  - 距离-曝光曲线扫描 (Distance-Exposure Profile)
  - 静态精度稳定性测试 (Static Accuracy & SNR)
  - 反射率一致性测试 (Reflectivity Test)
  - 平面度测试 (Planarity Test)
  - 深度/空间分辨力测试
- **数据记录**: 自动将测试结果（Mean, Std, SNR, Error 等）保存为 CSV 报表。

### 快速开始

1. **运行程序**
   ```bash
   python auto_test_tof_v2.py
   ```
   *程序会自动搜索并连接连接的 ToF 设备（优先尝试索引 0, 1）。*

2. **操作菜单**
   启动后将显示交互式菜单：
   ```text
   === ToF 自动化测试套件 v2.1 (精简版) ===
   1. 距离-曝光曲线扫描 (Distance-Exposure Profile)
   2. 静态精度稳定性测试 (Static Accuracy)
   3. 反射率测试 (Reflectivity Test)
   ...
   0. 退出 (Exit)
   ```

3. **常用快捷键 (在预览窗口中)**
   - `Space`: 确认/开始采集
   - `Q`: 取消/返回
   - `H`: 切换 HDR 模式
   - `[` / `]`: 调整 Depth Scale
   - `O` / `P`: 调整 Depth Offset

4. **输出文件**
   - 测试报告: `tof_test_report.csv`
   - 截图/Raw数据: 保存在当前目录下

## v4l2 (video4linux2)

```bash
# 识别设备
v4l2-ctl --list-devices

# example output:
# e-con's CX3 RDK with OV5640 (usb-0000:08:00.4-1.1):
#         /dev/video4
#         /dev/video5
#         /dev/media2

# CX3-UVC (usb-0000:08:00.4-1.2):
#         /dev/video2
#         /dev/video3
#         /dev/media1

# Integrated Camera: Integrated C (usb-0000:08:00.4-2.4):
#         /dev/video0
#         /dev/video1
#         /dev/media0
```


```bash

# 查看支持分辨率、格式
v4l2-ctl -d /dev/video2 --list-formats-ext

# example:
# $ v4l2-ctl -d /dev/video2 --list-formats-ext
# ioctl: VIDIOC_ENUM_FMT
#         Type: Video Capture

#         [0]: 'YUYV' (YUYV 4:2:2)
#                 Size: Discrete 320x720
#                         Interval: Discrete 0.033s (30.000 fps)
# $ v4l2-ctl -d /dev/video4 --list-formats-ext
# ioctl: VIDIOC_ENUM_FMT
#         Type: Video Capture

#         [0]: 'YUYV' (YUYV 4:2:2)
#                 Size: Discrete 1920x1080
#                         Interval: Discrete 0.033s (30.000 fps)
#                 Size: Discrete 640x480
#                         Interval: Discrete 0.017s (60.000 fps)
#                 Size: Discrete 1280x720
#                         Interval: Discrete 0.017s (60.000 fps)
#                 Size: Discrete 2592x1944
#                         Interval: Discrete 0.067s (15.000 fps)

# 获取推流
v4l2-ctl -d /dev/video2 --stream-mmap --stream-count=1 --stream-to=test.raw

# 可视化
ffplay -f video4linux2 -video_size 1920x1080 /dev/video4
```
