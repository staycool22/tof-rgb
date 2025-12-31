import cv2
import numpy as np
import time
import os
import csv
from datetime import datetime

from dataclasses import dataclass

@dataclass
class FrameInfo:
    frame_id: int
    timestamp: int
    text: int
    tint: int
    exposure_time: int

def decode_frame(frame_bytes, width, height):
    """
    解码一帧 OPN600x 数据。
    
    参数:
        frame_bytes: 输入的字节数据 (长度应为 3 * width * height * 2)
        width, height: 输出图像尺寸 (例如 320, 240)
        
    返回:
        raw_code: 原始深度数值 (Raw Code, 未标定)
        ir: 红外图 (uint16)
    """
    pixels = width * height
    expected = pixels * 3 * 2
    if len(frame_bytes) != expected:
        raise ValueError(f"缓冲区长度不匹配: {len(frame_bytes)} != {expected}")

    # 使用 numpy 进行高效解码
    raw = np.frombuffer(frame_bytes, dtype=">u2")  # 大端序读取
    words = raw.byteswap().astype(np.uint16)       # 转换为本地小端序
    
    # 重塑为 (height, 3, width)，其中每行包含 A, B, C 三个通道的数据
    rows = words.reshape(height, 3, width)
    A = rows[:, 0, :]
    B = rows[:, 1, :]
    C = rows[:, 2, :]

    # 原始数值解码逻辑:
    # 标准协议通常是 ((A | ((B >> 12) & 0x000F)) >> 2)
    # 恢复右移 2 位操作，使其符合标准协议。
    raw_code = ((A | ((B >> 12) & 0x000F)) >> 2).astype(np.uint16)
    
    # 红外数据提取
    ir = (C >> 4).astype(np.uint16)
    
    return raw_code, ir

# ==========================================
# 相机控制类
# ==========================================

class ToFCamera:
    """
    ToF 相机控制类，负责连接设备、设置参数、获取和解码数据。
    """
    def __init__(self, device_index=0):
        self.cap = None
        self.device_index = device_index
        self.target_width = 320
        self.target_height = 720
        self.current_width = 0
        self.current_height = 0
        self.hdr_state = 0 # 0: 关闭, 1: 开启
        self.current_exposure = 0 # 默认曝光值
        
        # 深度标定参数 (线性变换: Depth = Raw * Scale + Offset)
        self.depth_scale = 16.76  # 4.19 * 4
        self.depth_offset = 8.57

        # 强度校正参数 (用于修正高反/低反物体的距离偏差)
        self.ir_corr_slope = 1.50
        self.ir_corr_ref = 38.0  # 参考 IR 值
        
        self.connect()

    def get_recommended_settings(self, distance_mm):
        """
        根据目标距离推荐最佳曝光和标定参数。
        """
        # 全局统一使用稳健线性参数
        common_scale = 16.76 # 4.19 * 4
        common_offset = 8.57
        
        # 曝光策略: 保持 IR 强度在适中范围 (20-150)
        if distance_mm <= 150:
             exposure = -3
        elif distance_mm <= 250:
            exposure = -1
        elif distance_mm <= 350:
             exposure = 0
        elif distance_mm <= 450:
             exposure = 1
        elif distance_mm <= 650:
             exposure = 2
        else:
            exposure = 3 # 远距离需要更多光线
            
        return exposure, common_scale, common_offset

    def connect(self):
        """连接相机，尝试不同的后端 (MSMF, DSHOW)。"""
        print(f"正在连接相机索引 {self.device_index}...")
        
        # 优先尝试 MSMF (Windows 默认)
        self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_MSMF)
        if not self.cap.isOpened():
            print(f"索引 {self.device_index} (MSMF) 失败。尝试索引 0...")
            self.cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
        if not self.cap.isOpened():
            print("MSMF 失败。尝试 DSHOW...")
            self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        
        if not self.cap.isOpened():
            raise Exception("无法打开视频设备。")

        # 配置相机属性
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', '2'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0) # 禁用 RGB 转换
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # 1 = 手动曝光模式
        
        # 初始化设置
        self.set_exposure(self.current_exposure) 
        self.set_hdr(self.hdr_state)

        self.current_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.current_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"相机已打开: {self.current_width}x{self.current_height}")

    def get_frame(self):
        """
        获取一帧数据。
        返回: (depth_frame, ir_frame, depth_raw)
        """
        ret, frame = self.cap.read()
        if not ret:
            return None, None, None

        # 提取原始字节
        expected_bytes = 320 * 720 * 2
        raw_data = None

        try:
            if frame.nbytes >= expected_bytes:
                raw_data = frame.tobytes()[:expected_bytes]
        except Exception:
            pass

        depth_frame = None
        ir_frame = None
        depth_raw = None

        if raw_data is not None:
            try:
                # 解码
                depth_raw, ir_frame = decode_frame(raw_data, 320, 240)
                
                # 应用标定
                if self.depth_scale != 1.0 or self.depth_offset != 0:
                    depth_float = depth_raw.astype(np.float32) * self.depth_scale + self.depth_offset
                    
                    # 应用强度校正 (Intensity Correction)
                    if self.ir_corr_slope != 0:
                        ir_float = ir_frame.astype(np.float32)
                        
                        # 仅对低强度区域进行修正
                        mask = ir_float < self.ir_corr_ref
                        correction = np.zeros_like(ir_float)
                        correction[mask] = (ir_float[mask] - self.ir_corr_ref) * self.ir_corr_slope
                        
                        depth_float += correction
                    
                    depth_frame = np.clip(depth_float, 0, 65535).astype(np.uint16)
                else:
                    depth_frame = depth_raw.copy()
                    
            except Exception:
                pass

        return depth_frame, ir_frame, depth_raw

    def get_exposure_ms(self, value):
        """估算物理曝光时间 (ms)。"""
        estimated_us = (2 ** value) * 10000
        if estimated_us > 80000: estimated_us = 80000
        if estimated_us < 100: estimated_us = 100
        return estimated_us / 1000.0

    def set_exposure(self, value):
        """设置曝光等级。"""
        self.current_exposure = value
        self.cap.set(cv2.CAP_PROP_EXPOSURE, value)
        
        # 估算实际物理曝光时间
        estimated_ms = self.get_exposure_ms(value)
        print(f"设置曝光等级: {value} (估算物理曝光: {estimated_ms:.2f} ms)")

    def set_hdr(self, state):
        """设置 HDR 状态 (0=关闭, 1=开启)。"""
        self.hdr_state = state
        self.cap.set(cv2.CAP_PROP_CONTRAST, state)
        status_str = "开启" if state == 1 else "关闭"
        print(f"HDR 已{status_str} (Contrast={state})")

    def toggle_hdr(self):
        """切换 HDR 状态。"""
        self.set_hdr(1 - self.hdr_state)

    def release(self):
        """释放相机资源。"""
        if self.cap:
            self.cap.release()

# ==========================================
# 数据分析类
# ==========================================

class ToFAnalyzer:
    """
    提供各种静态方法用于计算 ROI 统计数据和指标。
    """
    @staticmethod
    def calculate_roi_stats(depth_stack, ir_stack, roi_rect, truth_mm=None, raw_stack=None):
        """
        计算 ROI 统计数据。
        """
        x, y, w, h = roi_rect
        roi_data = depth_stack[:, y:y+h, x:x+w]
        
        # 过滤有效像素 (0 < d < 4000)
        valid_mask = (roi_data > 0) & (roi_data < 4000)
        total_pixels = roi_data.size
        valid_pixels = np.sum(valid_mask)
        
        if valid_pixels == 0:
            return None

        # 1. 时域均值和标准差 (先沿时间轴计算)
        pixel_means = np.mean(roi_data, axis=0)
        pixel_stds = np.std(roi_data, axis=0)
        
        valid_pixel_mask = (pixel_means > 0) & (pixel_means < 4000)
        
        if np.sum(valid_pixel_mask) == 0:
             return None

        # 最终指标: 区域内有效像素的均值
        mean_val = np.mean(pixel_means[valid_pixel_mask])
        std_val = np.mean(pixel_stds[valid_pixel_mask]) 
        
        # 计算 Raw Mean (如果提供)
        raw_mean_val = 0
        if raw_stack is not None and len(raw_stack) > 0:
            roi_raw = raw_stack[:, y:y+h, x:x+w]
            pixel_means_raw = np.mean(roi_raw, axis=0)
            if np.any(valid_pixel_mask):
                raw_mean_val = np.mean(pixel_means_raw[valid_pixel_mask])

        # 2. 填充率
        fill_rate = (valid_pixels / total_pixels) * 100.0
        if fill_rate < 99.0:
            print(f"  [警告] Fill Rate {fill_rate:.2f}% < 99%")
        
        # 3. 信噪比
        snr = mean_val / std_val if std_val > 0.0001 else 0
        
        # 4. 红外强度
        ir_mean = 0
        if ir_stack is not None and len(ir_stack) > 0:
            roi_ir = ir_stack[:, y:y+h, x:x+w]
            ir_mean = np.mean(roi_ir)

        # 5. 误差
        error = 0
        rel_error = 0
        if truth_mm is not None:
            error = mean_val - truth_mm
            rel_error = (error / truth_mm) * 100.0 if truth_mm > 0 else 0
            
        return {
            "mean": mean_val,
            "raw_mean": raw_mean_val,
            "std": std_val,
            "fill_rate": fill_rate,
            "snr": snr,
            "ir_mean": ir_mean,
            "error": error,
            "rel_error": rel_error,
            "truth": truth_mm
        }

    @staticmethod
    def calculate_planarity_rmse(depth_frame, roi_rect):
        """
        计算平面度 RMSE (均方根误差)。
        拟合平面 Z = ax + by + c。
        """
        x, y, w, h = roi_rect
        roi = depth_frame[y:y+h, x:x+w]
        
        # 创建网格坐标
        Y, X = np.indices(roi.shape)
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        Z_flat = roi.flatten()
        
        valid = (Z_flat > 0) & (Z_flat < 4000)
        if np.sum(valid) < 10:
            return 0.0
            
        X_valid = X_flat[valid]
        Y_valid = Y_flat[valid]
        Z_valid = Z_flat[valid]
        
        # 拟合平面: A * [a, b, c].T = Z
        A = np.c_[X_valid, Y_valid, np.ones(X_valid.shape)]
        try:
            C, _, _, _ = np.linalg.lstsq(A, Z_valid, rcond=None)
            a, b, c = C
            Z_fit = a * X_valid + b * Y_valid + c
            return np.sqrt(np.mean((Z_valid - Z_fit)**2))
        except:
            return 0.0

# ==========================================
# 测试套件主类
# ==========================================

class TestSuite:
    def __init__(self):
        self.camera = ToFCamera()
        self.analyzer = ToFAnalyzer()
        self.log_file = "tof_test_report.csv"
        self._init_log_file()

    def _init_log_file(self):
        """初始化日志文件，写入表头。"""
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "TestType (测试类型)", 
                    "Timestamp (时间戳)", 
                    "Distance_Truth_mm (真实距离)", 
                    "Exposure_Level (曝光等级)", 
                    "Real_Exposure_ms (物理曝光)",
                    "Raw_Mean_Code (原始均值)",
                    "IR_Mean (红外均值)",
                    "Mean_mm (测量均值)", 
                    "StdDev_mm (标准差)", 
                    "FillRate_% (填充率)", 
                    "SNR (信噪比)", 
                    "Error_mm (误差)", 
                    "RelError_% (相对误差)", 
                    "Planarity_RMSE_mm (平面度RMSE)", 
                    "Calib_Scale (标定系数)",
                    "Calib_Offset (标定偏移)",
                    "Note (备注)", 
                    "HDR_State (HDR状态)"
                ])

    def run(self):
        """主运行循环，显示菜单。"""
        while True:
            print("\n=== ToF 自动化测试套件 v2.1 (精简版) ===")
            print("1. 距离-曝光曲线扫描 (Distance-Exposure Profile)")
            print("2. 静态精度稳定性测试 (Static Accuracy)")
            print("3. 反射率测试 (Reflectivity Test)")
            print("4. 平面度测试 (Planarity Test)")
            print("5. 实时预览与手动标定 (Live View)")
            print("6. 设置感兴趣区域 (Select ROI)")
            print("7. 深度分辨力测试 (Depth Resolution)")
            print("8. 空间分辨力测试 (Spatial Resolution)")
            print("9. 强度补偿校准 (Intensity Correction)")
            print("0. 退出 (Exit)")
            
            choice = input("请输入选项: ")
            
            if choice == '1':
                self.test_distance_exposure_profile()
            elif choice == '2':
                self.test_static_accuracy()
            elif choice == '3':
                self.test_reflectivity()
            elif choice == '4':
                self.test_planarity()
            elif choice == '5':
                self.live_view()
            elif choice == '6':
                self.select_roi_interactive()
            elif choice == '7':
                self.test_depth_resolution()
            elif choice == '8':
                self.test_spatial_resolution()
            elif choice == '9':
                self.test_calib_intensity_correction()
            elif choice == '0' or choice == 'q':
                break
            else:
                print("无效选择。")
        
        self.camera.release()

    # --- 辅助方法 ---

    def wait_for_key(self, prompt="按 Space 开始..."):
        print(f"\n{prompt}")
        print("操作: Space=开始, Q=取消, H=切换HDR, +/-=调整曝光")
        while True:
            d, i, _ = self.camera.get_frame()
            if d is not None:
                self.show_preview(d, i, extra_text=prompt)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 32: # Space
                cv2.destroyWindow("Test Preview")
                return True
            elif key == ord('q'):
                cv2.destroyWindow("Test Preview")
                return False
            elif key == ord('h'):
                self.camera.toggle_hdr()
            elif key == ord('+') or key == ord('='):
                self.camera.set_exposure(self.camera.current_exposure + 1)
            elif key == ord('-') or key == ord('_'):
                self.camera.set_exposure(self.camera.current_exposure - 1)

    def capture_stack(self, count=100, delay=0):
        stack = []
        ir_stack = []
        raw_stack = []
        print(f"正在采集 {count} 帧...")
        while len(stack) < count:
            d, i, raw_d = self.camera.get_frame()
            if d is not None:
                stack.append(d)
                if i is not None: ir_stack.append(i)
                if raw_d is not None: raw_stack.append(raw_d)
                self.show_preview(d, i, extra_text=f"Capturing {len(stack)}/{count}")
                if delay > 0: time.sleep(delay)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("采集已取消")
                break
        cv2.destroyWindow("Test Preview")
        return np.array(stack), np.array(ir_stack), np.array(raw_stack)

    def get_roi_rect(self):
        """获取当前 ROI (若未设置则返回中心默认值)。"""
        if hasattr(self, 'selected_roi') and self.selected_roi is not None:
            return self.selected_roi
        cx, cy = self.camera.current_width // 2, self.camera.current_height // 2
        w, h = 40, 40
        return (cx - w//2, cy - h//2, w, h)

    def select_roi_interactive(self):
        """交互式选择 ROI。"""
        print("\n[ROI 选择模式]")
        print("请用鼠标框选物体。Space/Enter=确认, R=重置, Q=取消。")

        while True:
            d, i, _ = self.camera.get_frame()
            if d is None: continue

            d_norm = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            d_display = cv2.cvtColor(d_norm, cv2.COLOR_GRAY2BGR)

            try:
                rect = cv2.selectROI("ROI Selection", d_display, showCrosshair=True, fromCenter=False)
                if rect[2] > 0 and rect[3] > 0:
                    print(f"ROI 已选择: {rect}")
                    cv2.destroyWindow("ROI Selection")
                    self.selected_roi = rect
                    return True
                else:
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord('q'):
                        cv2.destroyWindow("ROI Selection")
                        return False
                    elif key == ord('r'):
                        self.selected_roi = None
                        cv2.destroyWindow("ROI Selection")
                        return True
                    else:
                        cv2.destroyWindow("ROI Selection")
            except Exception:
                cv2.destroyAllWindows()
                return False

    def show_preview(self, depth, ir, extra_text=None):
        if depth is None: return
        d_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        d_display = cv2.cvtColor(d_norm, cv2.COLOR_GRAY2BGR)
        
        roi = self.get_roi_rect()
        x, y, w, h = roi
        cv2.rectangle(d_display, (x,y), (x+w, y+h), (0,255,0), 2)
        
        roi_depth = depth[y:y+h, x:x+w]
        valid_mask = (roi_depth > 0) & (roi_depth < 60000)
        center_val = int(np.mean(roi_depth[valid_mask])) if np.any(valid_mask) else 0
        
        hdr_txt = "HDR: ON" if self.camera.hdr_state else "HDR: OFF"
        cv2.putText(d_display, f"{center_val} mm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(d_display, hdr_txt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        if extra_text:
             cv2.putText(d_display, extra_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.imshow("Test Preview", d_display)

    def save_result(self, test_type, data, note=""):
        """保存测试结果到 CSV。"""
        exp_level = data.get("exposure", self.camera.current_exposure)
        try:
            real_exp_ms = f"{self.camera.get_exposure_ms(exp_level):.2f}"
        except:
            real_exp_ms = "Err"

        mean_mm = data.get('mean', 0)
        
        # 直接使用统计数据中的 Raw Mean
        raw_mean_code = data.get('raw_mean', 0)
            
        try:
            with open(self.log_file, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow([
                    test_type,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    data.get("truth", ""),
                    exp_level,
                    real_exp_ms,
                    f"{raw_mean_code:.2f}",
                    f"{data.get('ir_mean', 0):.1f}",
                    f"{mean_mm:.2f}",
                    f"{data.get('std', 0):.2f}",
                    f"{data.get('fill_rate', 0):.1f}",
                    f"{data.get('snr', 0):.1f}",
                    f"{data.get('error', 0):.2f}",
                    f"{data.get('rel_error', 0):.2f}",
                    f"{data.get('rmse', 0):.2f}",
                    f"{self.camera.depth_scale:.2f}",
                    f"{self.camera.depth_offset:.1f}",
                    note,
                    "ON" if self.camera.hdr_state else "OFF"
                ])
            print(f"数据已保存至 {self.log_file}")
        except PermissionError:
            print(f"\n[错误] 无法写入 {self.log_file}。请关闭文件后重试。")

    # --- 测试功能 ---

    def test_distance_exposure_profile(self):
        try:
            dist = float(input("输入真实距离 (mm): "))
            rec = self.camera.get_recommended_settings(dist)
            print(f"\n[推荐] Exp:{rec[0]}, Scale:{rec[1]}, Offset:{rec[2]}")
            if input("应用推荐设置? (y/n): ").lower() != 'n':
                self.camera.depth_scale = rec[1]
                self.camera.depth_offset = rec[2]
            
            start_exp = int(input("起始曝光: "))
            end_exp = int(input("结束曝光: "))
            step = int(input("步长: "))
        except ValueError:
            print("输入无效。")
            return

        if not self.wait_for_key("按 Space 开始扫描..."): return

        step_val = abs(step) * (1 if end_exp > start_exp else -1)
        # 包含结束值
        rng = range(start_exp, end_exp + (1 if step_val>0 else -1), step_val)

        for exp in rng:
            self.camera.set_exposure(exp)
            time.sleep(0.5)
            stack, ir_stack, raw_stack = self.capture_stack(100)
            stats = self.analyzer.calculate_roi_stats(stack, ir_stack, self.get_roi_rect(), dist, raw_stack)
            if stats:
                stats["exposure"] = exp
                print(f"Exp={exp}: Mean={stats['mean']:.1f}, Std={stats['std']:.1f}")
                self.save_result("Dist-Exp-Profile", stats)

    def test_static_accuracy(self):
        try:
            dist = float(input("输入真实距离 (mm): "))
            rec = self.camera.get_recommended_settings(dist)
            if input(f"推荐曝光 {rec[0]}，应用? (y/n): ").lower() != 'n':
                self.camera.set_exposure(rec[0])
                self.camera.depth_scale = rec[1]
                self.camera.depth_offset = rec[2]
        except ValueError: return

        if not self.wait_for_key("按 Space 采集静态数据..."): return

        stack, ir_stack, raw_stack = self.capture_stack(100)
        stats = self.analyzer.calculate_roi_stats(stack, ir_stack, self.get_roi_rect(), dist, raw_stack)
        if stats:
            print(f"\n均值: {stats['mean']:.2f}, 误差: {stats['error']:.2f}, Std: {stats['std']:.2f}, SNR: {stats['snr']:.2f}")
            self.save_result("Static-Accuracy", stats)

    def test_reflectivity(self):
        print("\n[反射率测试] 请依次放置 白/灰/黑 目标。")
        # 1. White
        if not self.select_roi_interactive(): return
        if not self.wait_for_key("放置白色目标，Space采集..."): return
        s_w, i_w, r_w = self.capture_stack(100)
        stats_w = self.analyzer.calculate_roi_stats(s_w, i_w, self.selected_roi, raw_stack=r_w)
        if not stats_w: return
        print(f"White Mean: {stats_w['mean']:.2f}")
        
        # 2. Grey
        if not self.select_roi_interactive(): return
        if not self.wait_for_key("放置灰卡，Space采集..."): return
        s_g, i_g, r_g = self.capture_stack(100)
        stats_g = self.analyzer.calculate_roi_stats(s_g, i_g, self.selected_roi, raw_stack=r_g)
        
        # 3. Black
        if not self.select_roi_interactive(): return
        if not self.wait_for_key("放置黑色目标，Space采集..."): return
        s_b, i_b, r_b = self.capture_stack(100)
        stats_b = self.analyzer.calculate_roi_stats(s_b, i_b, self.selected_roi, raw_stack=r_b)

        if stats_w and stats_g and stats_b:
            diff_wb = abs(stats_w['mean'] - stats_b['mean'])
            print(f"白黑差值: {diff_wb:.2f} mm")
            self.save_result("Reflectivity-White", stats_w, "White")
            self.save_result("Reflectivity-Grey", stats_g, "Grey")
            self.save_result("Reflectivity-Black", stats_b, f"Black, Diff={diff_wb:.2f}")

    def test_planarity(self):
        print("\n[平面度测试] 对准墙面。")
        if not self.wait_for_key(): return
        stack, _, _ = self.capture_stack(10)
        avg = np.mean(stack, axis=0)
        h, w = avg.shape
        roi_size = 100
        roi = ((w-roi_size)//2, (h-roi_size)//2, roi_size, roi_size)
        rmse = self.analyzer.calculate_planarity_rmse(avg, roi)
        print(f"RMSE: {rmse:.3f} mm")
        self.save_result("Planarity", {"rmse": rmse})

    def test_depth_resolution(self):
        print("\n[深度分辨力] 1.近处目标 2.远处目标")
        if not self.select_roi_interactive(): return
        print("采集 ROI 1 (近)...")
        s1, _, _ = self.capture_stack(50)
        stats1 = self.analyzer.calculate_roi_stats(s1, None, self.selected_roi)
        
        if not self.select_roi_interactive(): return
        print("采集 ROI 2 (远)...")
        s2, _, _ = self.capture_stack(50)
        stats2 = self.analyzer.calculate_roi_stats(s2, None, self.selected_roi)
        
        if stats1 and stats2:
            diff = abs(stats1['mean'] - stats2['mean'])
            noise = stats1['std'] + stats2['std']
            passed = diff > 3 * noise
            print(f"Diff: {diff:.2f}, NoiseSum: {noise:.2f}, Pass: {passed}")
            self.save_result("Depth-Res-ROI1", stats1, "Near")
            self.save_result("Depth-Res-ROI2", stats2, f"Far, Diff={diff:.2f}, Pass={passed}")

    def test_spatial_resolution(self):
        print("\n[空间分辨力] 截图保存。")
        if not self.wait_for_key("按 Space 拍照..."): return
        d, _, _ = self.camera.get_frame()
        if d is not None:
            ts = datetime.now().strftime('%H%M%S')
            cv2.imwrite(f"spatial_res_{ts}_vis.png", cv2.applyColorMap(cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U), cv2.COLORMAP_JET))
            cv2.imwrite(f"spatial_res_{ts}_raw.png", d)
            print(f"已保存截图: spatial_res_{ts}_*.png")
            self.save_result("Spatial-Res", {"mean":0}, f"Saved {ts}")

    def test_calib_intensity_correction(self):
        print("\n[强度补偿校准]")
        if not self.select_roi_interactive(): return
        if not self.wait_for_key("采集白色目标..."): return
        s_w, i_w, r_w = self.capture_stack(50)
        stats_w = self.analyzer.calculate_roi_stats(s_w, i_w, self.selected_roi, raw_stack=r_w)
        
        print("\n请保持距离不变，换黑色目标。")
        cv2.waitKey(0)
        if not self.select_roi_interactive(): return
        if not self.wait_for_key("采集黑色目标..."): return
        s_b, i_b, r_b = self.capture_stack(50)
        stats_b = self.analyzer.calculate_roi_stats(s_b, i_b, self.selected_roi, raw_stack=r_b)
        
        if stats_w and stats_b:
            dist_diff = stats_b['mean'] - stats_w['mean']
            ir_diff = stats_b['ir_mean'] - stats_w['ir_mean']
            if abs(ir_diff) > 1.0:
                slope = (stats_w['mean'] - stats_b['mean']) / (stats_b['ir_mean'] - stats_w['ir_mean'])
                print(f"Slope: {slope:.4f}")
                if input("应用参数? (y/n): ").lower() == 'y':
                    self.camera.ir_corr_slope = slope
                    self.camera.ir_corr_ref = stats_w['ir_mean']
            else:
                print("IR 差异太小。")

    def live_view(self):
        print("实时预览: S=截图, H=HDR, Q=退出, [/]=调整Scale, O/P=调整Offset")
        while True:
            d, i, raw_d = self.camera.get_frame()
            if d is None:
                if cv2.waitKey(10) & 0xFF == ord('q'): break
                continue
            
            # 显示逻辑
            d_norm = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            d_display = cv2.cvtColor(d_norm, cv2.COLOR_GRAY2BGR)
            
            roi = self.get_roi_rect()
            cv2.rectangle(d_display, (roi[0],roi[1]), (roi[0]+roi[2], roi[1]+roi[3]), (0,255,0), 2)
            
            roi_depth = d[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
            val = int(np.mean(roi_depth[(roi_depth>0)&(roi_depth<60000)])) if np.any(roi_depth) else 0
            
            cv2.putText(d_display, f"{val} mm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(d_display, f"Scale:{self.camera.depth_scale:.2f} Off:{self.camera.depth_offset}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.imshow("Live View", d_display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('s'): cv2.imwrite(f"live_{datetime.now().strftime('%H%M%S')}.png", d_display)
            elif key == ord('h'): self.camera.toggle_hdr()
            elif key == ord('['): self.camera.depth_scale = max(0.1, self.camera.depth_scale - 0.05)
            elif key == ord(']'): self.camera.depth_scale += 0.05
            elif key == ord('o'): self.camera.depth_offset -= 10
            elif key == ord('p'): self.camera.depth_offset += 10
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        suite = TestSuite()
        suite.run()
    except Exception as e:
        print(f"错误: {e}")
