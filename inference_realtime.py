# 根据自己需要调整 pads 和羽化 feather  的参数
import numpy as np
import cv2, os, sys, argparse, audio
import torch, face_detection
from models import Wav2Lip
import pyaudio
import collections
import time
import librosa

parser = argparse.ArgumentParser(description='实时唇形同步推理')

parser.add_argument('--checkpoint_path', type=str, 
                    help='模型检查点路径', required=True)

parser.add_argument('--face', type=str, 
                    help='输入视频/图片文件路径', required=True)

parser.add_argument('--device_index', type=int, default=None,
                    help='音频设备索引')

parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
                    help='人脸区域填充 (上, 下, 左, 右)')

parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], 
                    help='指定人脸边界框 (上, 下, 左, 右)')

parser.add_argument('--debug', action='store_true',
                    help='启用调试模式，显示音频处理信息')

args = parser.parse_args()
args.img_size = 384

# 检查输入文件类型
if os.path.isfile(args.face):
    file_ext = args.face.split('.')[-1].lower()
    args.static = file_ext in ['jpg', 'jpeg', 'png']
else:
    args.static = False

def face_detect(images):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
                                            flip_input=False, device=device)
    batch_size = 16  # 增加批处理大小
    
    # 预处理：缩放图像
    print("预处理：缩放图像...")
    scaled_images = []
    scale_factor = 0.5  # 缩放因子
    for img in images:
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        scaled_img = cv2.resize(img, (new_w, new_h))
        scaled_images.append(scaled_img)
    
    while 1:
        predictions = []
        try:
            total_batches = (len(scaled_images) + batch_size - 1) // batch_size
            for i in range(0, len(scaled_images), batch_size):
                batch_num = i // batch_size + 1
                print(f"\r人脸检测进度: {batch_num}/{total_batches} 批次", end="")
                predictions.extend(detector.get_detections_for_batch(np.array(scaled_images[i:i + batch_size])))
            print()  # 换行
        except RuntimeError:
            if batch_size == 1: 
                raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = args.pads
    total_faces = len(predictions)
    for idx, (rect, image) in enumerate(zip(predictions, images)):
        print(f"\r处理人脸: {idx + 1}/{total_faces}", end="")
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image)
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        # 将检测到的人脸坐标映射回原始图像尺寸
        scale = 1.0 / scale_factor
        y1 = max(0, int(rect[1] * scale) - pady1)
        y2 = min(image.shape[0], int(rect[3] * scale) + pady2)
        x1 = max(0, int(rect[0] * scale) - padx1)
        x2 = min(image.shape[1], int(rect[2] * scale) + padx2)
        
        results.append([x1, y1, x2, y2])
    print()  # 换行

    boxes = np.array(results)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    return results 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

class AudioProcessor:
    def __init__(self):
        """初始化音频处理器"""
        self.sample_rate = 16000  # 采样率 1s = 16000
        self.chunk_size = 640     # 1帧的mel 40ms 帧率25fps 16000/25=640
        self.channels = 1
        self.format = pyaudio.paFloat32
        self.stream = None
        self.audio_buffer = collections.deque(maxlen=16000)
        self.p = None
        self.is_running = False
        self.mel_step_size = 16   # mel频谱图的步长
        
    def start_stream(self, device_index=None):
        """启动音频流"""
        try:
            if self.p is None:
                self.p = pyaudio.PyAudio()
                
            if device_index is None:
                numdevices = self.p.get_host_api_info_by_index(0).get('deviceCount')
                for i in range(numdevices):
                    device_info = self.p.get_device_info_by_index(i)
                    if 'virtual' in device_info['name'].lower() or 'vb-audio' in device_info['name'].lower():
                        device_index = i
                        print(f"找到虚拟声卡设备: {device_info['name']}")
                        break
                        
            if device_index is None:
                print("未找到虚拟声卡设备，使用默认输入设备")
                device_index = self.p.get_default_input_device_info()['index']
                
            device_info = self.p.get_device_info_by_index(device_index)
            print(f"使用音频设备: {device_info['name']}")
            
            self.is_running = True
            
            self.stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            print("音频流启动成功")
            return self.stream
            
        except Exception as e:
            print(f"音频流启动失败: {e}")
            return None
            
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """音频回调函数"""
        try:
            if not self.is_running:
                return (in_data, pyaudio.paComplete)
                
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            self.audio_buffer.extend(audio_data)
            
            return (in_data, pyaudio.paContinue)
        except Exception as e:
            print(f"音频回调错误: {e}")
            return (in_data, pyaudio.paContinue)
            
    def read_audio(self):
        """读取音频数据并生成Mel频谱图"""
        try:
            if len(self.audio_buffer) < self.chunk_size:
                return None
                
            audio_data = np.array(list(self.audio_buffer)[-self.chunk_size:])
            audio_data = np.clip(audio_data, -1.0, 1.0)
            
            mel = audio.melspectrogram(audio_data)
            
            if mel.shape[1] < self.mel_step_size:
                pad_width = ((0, 0), (0, self.mel_step_size - mel.shape[1]))
                mel = np.pad(mel, pad_width, mode='edge')
            else:
                mel = mel[:, -self.mel_step_size:]
            
            return mel
            
        except Exception as e:
            print(f"读取音频数据错误: {e}")
            return None
            
    def stop_stream(self):
        """停止音频流"""
        try:
            self.is_running = False
            if self.stream is not None:
                self.stream.stop_stream()
                self.stream.close()
            if self.p is not None:
                self.p.terminate()
            print("音频流已停止")
        except Exception as e:
            print(f"停止音频流错误: {e}")
            
    def __del__(self):
        """析构函数"""
        self.stop_stream()

class LipSyncModel:
    def __init__(self, checkpoint_path):
        """初始化唇形同步模型"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.face_detector = None
        self.load_model(checkpoint_path)
        self.load_face_detector()
        
        # 加载蒙版图像
        mask_path = os.path.join('.', 'models', 'mask.png')
        self.mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if self.mask_img is None:
            raise ValueError("找不到蒙版图像: " + mask_path)
            
        # 性能监控
        self.process_times = collections.deque(maxlen=100)
        self.inference_times = collections.deque(maxlen=100)
        
    def load_model(self, checkpoint_path):
        """加载模型"""
        try:
            print("加载Wav2Lip模型...")
            self.model = Wav2Lip()
            
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"找不到检查点文件: {checkpoint_path}")
                
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            self.model.load_state_dict(checkpoint['state_dict'])
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print("模型加载成功")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
            
    def load_face_detector(self):
        """加载人脸检测器"""
        try:
            print("加载人脸检测器...")
            self.face_detector = face_detection.FaceAlignment(
                face_detection.LandmarksType._2D,
                flip_input=False,
                device=self.device if isinstance(self.device, str) else 'cuda' if torch.cuda.is_available() else 'cpu'
            )
            print("人脸检测器加载成功")
        except Exception as e:
            print(f"人脸检测器加载失败: {e}")
            raise
        
    def process_frame(self, frame, mel, face_coords):
        """处理单帧"""
        try:
            if frame is None:
                return frame
                
            # 记录开始时间
            start_time = time.time()
            
            # 准备输入数据
            y1, y2, x1, x2 = face_coords
            
            face = frame[y1:y2, x1:x2]
            face = cv2.resize(face, (args.img_size, args.img_size))
            
            # 创建批处理数据
            img_batch = np.array([face])
            img_masked = img_batch.copy()
            img_masked[:, args.img_size//2:] = 0
            
            # 准备模型输入
            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            
            # 转换为tensor
            img_batch = torch.from_numpy(np.transpose(img_batch, (0, 3, 1, 2))).float().to(self.device)
            
            # 记录推理开始时间
            inference_start = time.time()
            
            # 模型推理
            with torch.amp.autocast('cuda' if self.device == 'cuda' else 'cpu'):
                with torch.no_grad():
                    mel_batch = np.array([mel])
                    mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
                    mel_batch = torch.from_numpy(np.transpose(mel_batch, (0, 3, 1, 2))).float().to(self.device)
                    pred = self.model(mel_batch, img_batch)
            
            # 记录推理时间
            inference_time = time.time() - inference_start
            self.inference_times.append(inference_time)
            
            # 后处理
            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
            p = pred[0]
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            
            # 使用蒙版进行融合
            resized_mask = cv2.resize(self.mask_img, (x2 - x1, y2 - y1))
            _, binary_mask = cv2.threshold(resized_mask, 127, 255, cv2.THRESH_BINARY)
            binary_mask = binary_mask.astype(np.uint8)
            dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
            feather_radius = 15  # 减小羽化半径，让嘴型变化更明显
            alpha = np.clip(dist / feather_radius, 0, 1)
            alpha = cv2.GaussianBlur(alpha, (5,5), 0)  # 减小高斯模糊核大小
            
            # 融合图像
            original_face = frame[y1:y2, x1:x2].astype(np.float32)
            generated_face = p.astype(np.float32)
            
            blended_face = (alpha[..., None] * generated_face + (1 - alpha[..., None]) * original_face).astype(np.uint8)
            
            # 将处理后的面部区域放回原图
            output = frame.copy()
            output[y1:y2, x1:x2] = blended_face
            
            # 记录总处理时间
            total_time = time.time() - start_time
            self.process_times.append(total_time)
            
            return output
            
        except Exception as e:
            print(f"处理帧错误: {e}")
            return frame
            
    def get_performance_stats(self):
        """获取性能统计信息"""
        if not self.process_times or not self.inference_times:
            return "无性能数据"
            
        avg_process = sum(self.process_times) / len(self.process_times)
        avg_inference = sum(self.inference_times) / len(self.inference_times)
        
        return f"处理时间: {avg_process*1000:.1f}ms, 推理: {avg_inference*1000:.1f}ms"

    def __del__(self):
        """清理资源"""
        pass

def get_face_coords_cache_path(video_path):
    """获取人脸位置缓存文件路径"""
    video_dir = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(video_dir, f"{video_name}.npy")

def save_face_coords(coords, cache_path):
    """保存人脸位置到缓存文件"""
    try:
        np.save(cache_path, coords)
        print(f"人脸位置已保存到: {cache_path}")
    except Exception as e:
        print(f"保存人脸位置失败: {e}")

def load_face_coords(cache_path):
    """从缓存文件加载人脸位置"""
    try:
        if os.path.exists(cache_path):
            coords = np.load(cache_path)
            print(f"从缓存加载人脸位置: {cache_path}")
            return coords
        else:
            print(f"缓存文件不存在: {cache_path}")
    except Exception as e:
        print(f"加载人脸位置失败: {e}")
    return None

def main():
    print("初始化唇形同步模型...")
    model = LipSyncModel(args.checkpoint_path)
    
    print("初始化音频处理器...")
    audio_processor = AudioProcessor()
    
    print("\n启动音频流...")
    audio_stream = audio_processor.start_stream(args.device_index)
    if audio_stream is None:
        print("无法启动音频流")
        return
        
    print("读取输入文件...")
    if args.static:
        frame = cv2.imread(args.face)
        if frame is None:
            print("无法读取图片文件")
            return
        print("成功读取图片文件")
        frames = [frame]
        target_fps = 25  # 静态图片使用25fps
    else:
        cap = cv2.VideoCapture(args.face)
        if not cap.isOpened():
            print("无法打开视频文件")
            return
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"视频信息:")
        print(f"- 分辨率: {frame_width}x{frame_height}")
        print(f"- 帧率: {fps}")
        print(f"- 总帧数: {total_frames}")
    
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        print(f"成功读取 {len(frames)} 帧")
        
        print("创建倒放循环...")
        original_frames = frames.copy()
        reverse_frames = frames[::-1]
        frames = original_frames + reverse_frames
        print(f"倒放循环创建完成，总帧数: {len(frames)}")
        target_fps = fps  # 使用输入视频的原始帧率
    
    print("\n预处理：检测人脸位置...")
    cache_path = get_face_coords_cache_path(args.face)
    face_coords = load_face_coords(cache_path)
    
    if face_coords is None:
        print("开始人脸检测...")
        if args.box[0] == -1:
            face_det_results = face_detect(frames)
        else:
            print('使用指定的边界框...')
            y1, y2, x1, x2 = args.box
            face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]
        
        face_coords = np.array([coords for _, coords in face_det_results])
        save_face_coords(face_coords, cache_path)
    else:
        print("使用缓存的人脸位置数据")
        if not args.static:
            print("创建人脸位置倒放循环...")
            original_length = len(face_coords) // 2
            face_coords = np.concatenate([
                face_coords[:original_length],
                face_coords[:original_length][::-1]
            ])
            print("人脸位置倒放循环创建完成")
    
    print("人脸位置检测完成")
    
    print("\n开始主循环...")
    print("按 '+' 键放大窗口")
    print("按 '-' 键缩小窗口")
    
    last_time = time.time()
    fps_update_interval = 0.5
    fps_buffer = collections.deque(maxlen=10)
    last_stats_time = time.time()
    frames_processed = 0  # 帧处理索引
    
    display_width = 540
    display_height = 960
    
    cv2.namedWindow('Wav2Lip', cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.setWindowProperty('Wav2Lip', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.resizeWindow('Wav2Lip', display_width, display_height)
    
    frame_interval = 1.0 / target_fps  # 计算每帧之间的时间间隔 
    next_frame_time = time.time()  # 下一帧的预期时间
    
    while True:
        try:
            for i, current_frame in enumerate(frames):
                # 等待直到达到下一帧的时间
                current_time = time.time()
                if current_time < next_frame_time:
                    sleep_time = next_frame_time - current_time
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                
                mel = audio_processor.read_audio()  # 获取1帧音频数据 mel 频谱
                output = model.process_frame(current_frame, mel, face_coords[i % len(face_coords)])
                
                # 计算当前帧在原始视频中的位置
                original_frame_count = i % (len(frames) // 2)
                is_reverse = i >= len(frames) // 2
                
                # 在画面上显示帧信息
                frame_info = f"Frame: {original_frame_count}/{len(frames)//2} {'(倒放)' if is_reverse else ''}"
                
                frames_processed += 1
                current_time = time.time()
                elapsed = current_time - last_time
                
                if elapsed >= fps_update_interval:
                    current_fps = frames_processed / elapsed
                    fps_buffer.append(current_fps)
                    avg_fps = sum(fps_buffer) / len(fps_buffer)
                    
                    if current_time - last_stats_time >= 2.0:
                        stats = model.get_performance_stats()
                        if args.debug:
                            print(f"FPS: {avg_fps:.1f} | {stats} | {frame_info}")
                        else:
                            print(f"FPS: {avg_fps:.1f} | {stats} | {frame_info}")
                        last_stats_time = current_time
                    else:
                        if args.debug:
                            print(f"FPS: {avg_fps:.1f} | {frame_info}")
                        else:
                            print(f"FPS: {avg_fps:.1f} | {frame_info}")
                        
                    frames_processed = 0
                    last_time = current_time
                
                cv2.imshow('Wav2Lip', output)
                
                # 更新下一帧的时间
                next_frame_time += frame_interval
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('+'):
                    display_width = int(display_width * 1.1)
                    display_height = int(display_width * frames[0].shape[0] / frames[0].shape[1])
                    cv2.resizeWindow('Wav2Lip', display_width, display_height)
                elif key == ord('-'):
                    display_width = int(display_width * 0.9)
                    display_height = int(display_width * frames[0].shape[0] / frames[0].shape[1])
                    cv2.resizeWindow('Wav2Lip', display_width, display_height)
                    
        except Exception as e:
            print(f"主循环错误: {e}")
            break
            
    print("清理资源...")
    cv2.destroyAllWindows()
    audio_processor.stop_stream()
    print("程序结束")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n程序出错: {str(e)}")
        import traceback
        print("详细错误信息:")
        print(traceback.format_exc())
    finally:
        print("程序完全退出")