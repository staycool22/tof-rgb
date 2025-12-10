# tof-rgb

## usage

```bash
python tof_rgb.py /dev/tof /dev/rgb
# example:
# python tof_rgb.py /dev/video2 /dev/video4
```

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