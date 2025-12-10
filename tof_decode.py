"""
Image frame decoder (Python version)

Implements the same decoding as
`OPN600x_ARM_Linux_SDK_v2.1.0.13/src/imge_process/image_process.cpp:25-87`.

Input layout per frame (big-endian 16-bit words): for each image row i,
the row contains A_row[i][0..width-1], then B_row[i][0..width-1], then
 C_row[i][0..width-1]. Overall buffer length is 3*width*height*2 bytes.

Decoding logic:
  depth  = ((A | ((B >> 12) & 0x000F)) >> 2)
  ir     = (C >> 4)
  status = ((B >> 10) & 0x0003)
  bk_ir  = ((B >> 4)  & 0x003F)

FrameInfo fields from last row of IR:
  frame_id      = ir[-1,0] + ir[-1,1] * 4096
  timestamp     = ir[-1,2] + ir[-1,3] * 4096
  text          = ir[-1,4] - 43
  tint          = ir[-1,5] - 43
  exposure_time = ir[-1,6] | (ir[-1,7] << 12)

Provides a NumPy implementation; if NumPy is unavailable, a pure-Python
decoder is provided.
"""

from dataclasses import dataclass
from typing import Tuple, Optional

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore


@dataclass
class FrameInfo:
    frame_id: int
    timestamp: int
    text: int
    tint: int
    exposure_time: int


def _bswap16_py(x: int) -> int:
    return ((x >> 8) & 0x00FF) | ((x & 0x00FF) << 8)


def decode_frame(frame: np.uint8, width: int, height: int):
    """
    Decode one frame from `buf` using the OPN600x image decode logic.

    Parameters
    - buf: bytes, big-endian 16-bit words, length == 3*width*height*2
    - width, height: frame dimensions

    Returns
    - depth: np.ndarray[uint16] of shape (height, width)
    - ir: np.ndarray[uint16] of shape (height, width)
    - status: np.ndarray[uint8] of shape (height, width)
    - bk_ir: np.ndarray[uint8] of shape (height, width)
    - info: FrameInfo

    Raises
    - ValueError on invalid arguments or size mismatch
    """
    buf = frame.tobytes()

    if width <= 0 or height <= 0:
        raise ValueError("width/height must be positive")
    pixels = width * height
    expected = pixels * 3 * 2
    if len(buf) != expected:
        raise ValueError(f"buffer length mismatch: {len(buf)} != {expected}")

    if np is not None:
        raw = np.frombuffer(buf, dtype=">u2")  # big-endian
        words = raw.byteswap().astype(np.uint16)  # to native little-endian
        # reshape as (height, 3, width) where columns are A,B,C rows per line
        rows = words.reshape(height, 3, width)
        A = rows[:, 0, :]
        B = rows[:, 1, :]
        C = rows[:, 2, :]

        depth = ((A | ((B >> 12) & 0x000F)) >> 2).astype(np.uint16)
        ir = (C >> 4).astype(np.uint16)
        status = ((B >> 10) & 0x0003).astype(np.uint8)
        bk_ir = ((B >> 4) & 0x003F).astype(np.uint8)

        base = height - 1
        info = FrameInfo(
            frame_id=int(ir[base, 0] + ir[base, 1] * 4096),
            timestamp=int(ir[base, 2] + ir[base, 3] * 4096),
            text=int(ir[base, 4]) - 43,
            tint=int(ir[base, 5]) - 43,
            exposure_time=int(ir[base, 6] | (ir[base, 7] << 12)),
        )
        return depth, ir, status, bk_ir, info

    # Fallback pure-Python implementation
    import struct

    words = list(struct.unpack(f">{pixels*3}H", buf))
    # convert to little-endian values
    words = [((w >> 8) | ((w & 0xFF) << 8)) for w in words]

    depth = [[0] * width for _ in range(height)]
    ir = [[0] * width for _ in range(height)]
    status = [[0] * width for _ in range(height)]
    bk_ir = [[0] * width for _ in range(height)]

    for i in range(height):
        for j in range(width):
            a = words[i * 3 * width + j]
            b = words[(i * 3 + 1) * width + j]
            c = words[(i * 3 + 2) * width + j]
            depth[i][j] = ((a | ((b >> 12) & 0x000F)) >> 2) & 0xFFFF
            ir[i][j] = (c >> 4) & 0xFFFF
            status[i][j] = ((b >> 10) & 0x0003) & 0xFF
            bk_ir[i][j] = ((b >> 4) & 0x003F) & 0xFF

    base = height - 1
    info = FrameInfo(
        frame_id=ir[base][0] + ir[base][1] * 4096,
        timestamp=ir[base][2] + ir[base][3] * 4096,
        text=ir[base][4] - 43,
        tint=ir[base][5] - 43,
        exposure_time=(ir[base][6] | (ir[base][7] << 12)),
    )
    return depth, ir, status, bk_ir, info


def _example():
    # Construct synthetic frame identical to C example
    w, h = 8, 2
    import struct
    words = [0] * (h * 3 * w)
    for i in range(h):
        for j in range(w):
            a = (0x3C00 | (j & 0x0F)) & 0xFFFF
            b = ((0x2 << 10) | (0x15 << 4) | (0xA << 12)) & 0xFFFF
            c = (0xABC0 | (j & 0x0F)) & 0xFFFF
            words[i * 3 * w + j] = a
            words[(i * 3 + 1) * w + j] = b
            words[(i * 3 + 2) * w + j] = c
    # embed frame info in last row IR (C plane stores IR << 4)
    last_row = h - 1
    vals = [1234, 5, 42, 7, 50 + 43, 60 + 43, 0x345, 0x2]
    for k, v in enumerate(vals):
        words[(last_row * 3 + 2) * w + k] = (v << 4) & 0xFFFF
    # convert to big-endian bytes
    buf = struct.pack(f">{len(words)}H", *words)

    d, i, s, b, info = decode_frame(buf, w, h)
    print("frame_info:", info)
    if np is not None:
        print("depth[0,0]", d[0, 0], "ir[0,0]", i[0, 0], "status[0,0]", s[0, 0], "bk_ir[0,0]", b[0, 0])
    else:
        print("depth[0,0]", d[0][0], "ir[0,0]", i[0][0], "status[0,0]", s[0][0], "bk_ir[0,0]", b[0][0])


def query_v4l2_resolution(device: str = "/dev/video0"):
    """
    Query current v4l2 capture resolution for the given device.

    Tries `v4l2-ctl --get-fmt` first; falls back to opening with OpenCV
    briefly to read `CAP_PROP_FRAME_WIDTH/HEIGHT`.
    Returns `(width, height)`.
    """
    import subprocess, re
    try:
        out = subprocess.check_output(["v4l2-ctl", "--device", device, "--get-fmt"], stderr=subprocess.STDOUT, text=True)
        m = re.search(r"Width/Height\s+(\d+)\s*/\s*(\d+)", out)
        if m:
            return int(m.group(1)), int(m.group(2))
    except Exception:
        pass
    try:
        import cv2
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {device} to query resolution")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return w, h
    except Exception as e:
        raise RuntimeError(f"failed to query v4l2 resolution: {e}")


def open_v4l2_device(device: str = "/dev/video0"):
    """
    Open v4l2 device after verifying resolution is exactly 320x720.

    Returns an opened `cv2.VideoCapture` handle.
    Raises `ValueError` if resolution mismatches; `RuntimeError` if open fails.
    """
    import cv2
    w, h = query_v4l2_resolution(device)
    if (w, h) != (320, 720):
        raise ValueError(f"expected 320x720, got {w}x{h} on {device}")
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {device}")
    return cap


if __name__ == "__main__":
    import sys
    import numpy as np
    import cv2
    import time

    if len(sys.argv) >= 4 and sys.argv[1].startswith("/dev/video"):
        device = sys.argv[1]
        w = int(sys.argv[2]); h = int(sys.argv[3])
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'Y16 '))
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        if not cap.isOpened():
            raise SystemExit(f"failed to open {device}")
        time.sleep(0.5)
        ret, frame = cap.read()
        if not ret:
            raise SystemExit("capture failed")
        
        try:
            depth0, ir0, status0, bk0, info0 = decode_frame(frame, w, h)
        except Exception as e:
            print(f"decode failed: {e}")
            sys.exit(1)
        
        t0 = time.time()
        k = np.ones((3,3), dtype=np.float32) / 9.0
        for _ in range(10000):
            _ = cv2.filter2D(depth0.astype(np.float32), -1, k, borderType=cv2.BORDER_REPLICATE)
        t1 = time.time()
        print("start tv_sec:{}, tv_usec:{}".format(int(t0), int((t0-int(t0))*1e6)))
        print("end tv_sec:{}, tv_usec:{}".format(int(t1), int((t1-int(t1))*1e6)))
        region = depth0[120:130, 160:170]
        depth_tmp = int(region.mean())
        print("depth = {}".format(depth_tmp))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            try:
                depth, ir, status, bk_ir, info = decode_frame(frame, w, h)
                print(info)
            except Exception as e:
                print(f"decode failed: {e}")
                break
            d16 = depth.copy(); i16 = ir.copy(); b8 = bk_ir.copy()
            d16 = (d16 - d16.min())
            d16 = (d16 / (d16.max() + 1e-6) * 255.0).astype(np.uint8)
            i16 = (i16 - i16.min())
            i16 = (i16 / (i16.max() + 1e-6) * 255.0).astype(np.uint8)
            b8_vis = (b8.astype(np.uint16) * 4).clip(0, 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(d16, cv2.COLORMAP_JET)
            cv2.imshow("depth", depth_color)
            cv2.imshow("IR", i16)
            cv2.imshow("BK_IR", b8_vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Usage: python3 decode.py /dev/videoX 320 240")