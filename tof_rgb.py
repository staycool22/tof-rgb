import cv2
import numpy as np
import sys
import time
import json
import threading

from tof_decode import decode_frame

"""
TOF and RGB device open helpers.

Provides functions to open ToF v4l2 device (raw 16-bit) and a separate
RGB camera device for color frames.
"""

class TofRgbCamera:
    def __init__(self, tof_device, rgb_device, config, use_threading=True):
        self.use_threading = use_threading

        # TOF Camera setup
        self.tof_cap = cv2.VideoCapture(tof_device, cv2.CAP_V4L2)
        if not self.tof_cap.isOpened():
            raise RuntimeError(f"failed to open tof device {tof_device}")
        
        tof_config = config['tof']
        self.tof_cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        self.tof_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'Y16 '))
        self.tof_cap.set(cv2.CAP_PROP_FRAME_WIDTH, tof_config['width'])
        self.tof_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, tof_config['height'])
        self.tof_w = tof_config['width']
        self.tof_h = tof_config['height']
        
        if self.use_threading:
            self.tof_ret, self.tof_frame = self.tof_cap.read()
            self.tof_lock = threading.Lock()
            self.tof_thread = threading.Thread(target=self._update_tof, args=())
            self.tof_thread.daemon = True
            self.tof_thread.start()

        # RGB Camera setup
        self.rgb_cap = cv2.VideoCapture(rgb_device, cv2.CAP_V4L2)
        if not self.rgb_cap.isOpened():
            raise RuntimeError(f"failed to open rgb device {rgb_device}")

        rgb_config = config['rgb']
        self.rgb_cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
        self.rgb_cap.set(cv2.CAP_PROP_FRAME_WIDTH, rgb_config['width'])
        self.rgb_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, rgb_config['height'])
        self.rgb_cap.set(cv2.CAP_PROP_FPS, rgb_config['fps'])
        self.rgb_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # controls = rgb_config.get('controls', {})
        # self.rgb_cap.set(cv2.CAP_PROP_BRIGHTNESS, controls.get('brightness', 0))
        # self.rgb_cap.set(cv2.CAP_PROP_CONTRAST, controls.get('contrast', 6))
        # self.rgb_cap.set(cv2.CAP_PROP_SATURATION, controls.get('saturation', 4))
        # self.rgb_cap.set(cv2.CAP_PROP_HUE, controls.get('hue', 0))
        # self.rgb_cap.set(cv2.CAP_PROP_AUTOWB, 1 if controls.get('white_balance_automatic', True) else 0)
        # if not controls.get('white_balance_automatic', True):
        #     self.rgb_cap.set(cv2.CAP_PROP_WB_TEMPERATURE, controls.get('white_balance_temperature', 2))
        # self.rgb_cap.set(cv2.CAP_PROP_SHARPNESS, controls.get('sharpness', 1))
        # self.rgb_cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, controls.get('auto_exposure', 3))
        # if controls.get('auto_exposure') == 1: # Manual mode
        #     self.rgb_cap.set(cv2.CAP_PROP_EXPOSURE, controls.get('exposure_time_absolute', 20))
        # self.rgb_cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if controls.get('focus_automatic_continuous', True) else 0)
        # if not controls.get('focus_automatic_continuous', True):
        #     self.rgb_cap.set(cv2.CAP_PROP_FOCUS, controls.get('focus_absolute', 0))

        if self.use_threading:
            self.rgb_ret, self.rgb_frame = self.rgb_cap.read()
            self.rgb_lock = threading.Lock()
            self.rgb_thread = threading.Thread(target=self._update_rgb, args=())
            self.rgb_thread.daemon = True
            self.rgb_thread.start()

    def _update_tof(self):
        while True:
            ret, frame = self.tof_cap.read()
            with self.tof_lock:
                self.tof_ret = ret
                self.tof_frame = frame

    def _update_rgb(self):
        while True:
            ret, frame = self.rgb_cap.read()
            with self.rgb_lock:
                self.rgb_ret = ret
                self.rgb_frame = frame

    def get_rgb_image(self):
        if self.use_threading:
            with self.rgb_lock:
                return self.rgb_ret, self.rgb_frame.copy()
        else:
            return self.rgb_cap.read()

    def get_tof_image(self):
        if self.use_threading:
            with self.tof_lock:
                ret, frame = self.tof_ret, self.tof_frame.copy()
        else:
            ret, frame = self.tof_cap.read()
            
        if ret:
            depth, ir, status, bk_ir, info = decode_frame(frame, self.tof_w, self.tof_h)
            return ret, depth, ir
        return ret, None, None

    def release(self):
        self.tof_cap.release()
        self.rgb_cap.release()
        

if __name__ == "__main__":
    with open('config.json', 'r') as f:
        config = json.load(f)

    if len(sys.argv) >= 3 and sys.argv[1].startswith("/dev/video") and sys.argv[2].startswith("/dev/video") :
        camera = TofRgbCamera(sys.argv[1], sys.argv[2], config)
        time.sleep(0.5)

        while True:
            ret_tof, depth, ir = camera.get_tof_image()
            ret_rgb, frame_rgb = camera.get_rgb_image()

            if ret_tof:
                d16 = depth.copy()
                d16 = (d16 - d16.min())
                d16 = (d16 / (d16.max() + 1e-6) * 255.0).astype(np.uint8)
                depth_color = cv2.applyColorMap(d16, cv2.COLORMAP_JET)
                cv2.imshow("Depth_Color", depth_color)

                i16 = ir.copy()
                i16 = (i16 - i16.min())
                i16 = (i16 / (i16.max() + 1e-6) * 255.0).astype(np.uint8)
                cv2.imshow("IR", i16)
            else:
                print("ToF capture failed")

            if ret_rgb:
                cv2.imshow("RGB Frame", frame_rgb)
            else:
                print("RGB capture failed")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        camera.release()
        cv2.destroyAllWindows()