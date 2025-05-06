from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import uuid
import shutil
import threading
from datetime import datetime
import subprocess
import argparse
from os import listdir, path
import numpy as np
import scipy, cv2, audio
import json
import platform
import torch, face_detection
from torch.cuda.amp import autocast, GradScaler
from apscheduler.schedulers.background import BackgroundScheduler

#===============推理代码部分=================#

app = Flask(__name__)

# 配置上传文件夹和允许文件格式
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'mov', 'avi', 'wav', 'jpg', 'png', 'jpeg'}
app.config['CHECKPOINT_PATH'] = 'final_checkpionts/checkpoint_step001320000.pth'  # 改进后的模型路径配置

# 创建上传文件夹
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

#===============推理代码部分=================#

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str, 
                    help='Name of saved checkpoint to load weights from', required=True)
parser.add_argument('--face', type=str, 
                    help='Filepath of video/image that contains faces to use', required=True)
parser.add_argument('--audio', type=str, 
                    help='Filepath of video/audio file to use as raw audio source', required=True)
parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.', 
                    default='results/result_voice.mp4')
parser.add_argument('--static', type=bool, 
                    help='If True, then use only first video frame for inference', default=False)
parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', 
                    default=25., required=False)
parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
                    help='Padding (top, bottom, left, right). Please adjust to include chin at least')
parser.add_argument('--face_det_batch_size', type=int, 
                    help='Batch size for face detection', default=4)
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=8)
parser.add_argument('--resize_factor', default=1, type=int, 
                    help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')
parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], 
                    help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. '
                         'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')
parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], 
                    help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
                         'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')
parser.add_argument('--rotate', default=False, action='store_true',
                    help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
                         'Use if you get a flipped result, despite feeding a normal looking video')
parser.add_argument('--nosmooth', default=False, action='store_true',
                    help='Prevent smoothing face detections over a short temporal window')

args = None  # 将在API调用时初始化

def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes

def face_detect(images):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
                                            flip_input=False, device=device)
    batch_size = args.face_det_batch_size
    while True:
        predictions = []
        try:
            for i in range(0, len(images), batch_size):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1: 
                raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = args.pads
    for rect, image in zip(predictions, images):
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image)
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')
        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        results.append([x1, y1, x2, y2])
    
    boxes = np.array(results)
    if not args.nosmooth: 
        boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1:y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]
    del detector
    return results 

def datagen(frames, mels):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
    if args.box[0] == -1:
        if not args.static:
            face_det_results = face_detect(frames)
        else:
            face_det_results = face_detect([frames[0]])
    else:
        print('Using the specified bounding box instead of face detection...')
        y1, y2, x1, x2 = args.box
        face_det_results = [[f[y1:y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

    for i, m in enumerate(mels):
        idx = 0 if args.static else i % len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()
        face = cv2.resize(face, (args.img_size, args.img_size))
        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)
        if len(img_batch) >= args.wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
            img_masked = img_batch.copy()
            img_masked[:, args.img_size//2:] = 0
            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
        img_masked = img_batch.copy()
        img_masked[:, args.img_size//2:] = 0
        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
        yield img_batch, mel_batch, frame_batch, coords_batch

mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(path):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    model = model.to(device)
    return model.eval()

def wav2lip_main(api_args):
    global args
    args = api_args
    args.img_size = 384

    if not os.path.isfile(args.face):
        raise ValueError('--face argument must be a valid path to video/image file')
    elif args.face.split('.')[1].lower() in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(args.face)]
        fps = args.fps
    else:
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        print('Reading video frames...')
        full_frames = []
        while True:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if args.resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1] // args.resize_factor, frame.shape[0] // args.resize_factor))
            if args.rotate:
                frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
            y1, y2, x1, x2 = args.crop
            if x2 == -1: 
                x2 = frame.shape[1]
            if y2 == -1: 
                y2 = frame.shape[0]
            frame = frame[y1:y2, x1:x2]
            full_frames.append(frame)
            reverse_frames = full_frames[::-1]  # 倒放循环
        full_frames = full_frames + reverse_frames + full_frames  # 倒放循环

    print("Number of frames available for inference: " + str(len(full_frames)))
    if not args.audio.endswith('.wav'):
        print('Extracting raw audio...')
        command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp/temp.wav')
        subprocess.call(command, shell=True)
        args.audio = 'temp/temp.wav'

    wav = audio.load_wav(args.audio, 16000)
    mel = audio.melspectrogram(wav)
    print(mel.shape)
    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    mel_chunks = []
    mel_idx_multiplier = 80. / fps 
    i = 0
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1

    print("Length of mel chunks: {}".format(len(mel_chunks)))
    full_frames = full_frames[:len(mel_chunks)]
    batch_size = args.wav2lip_batch_size
    gen = datagen(full_frames.copy(), mel_chunks)

    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, total=int(np.ceil(float(len(mel_chunks)) / batch_size)))):
        if i == 0:
            model = load_model(args.checkpoint_path)
            print("Model loaded")
            frame_h, frame_w = full_frames[0].shape[:-1]
            out = cv2.VideoWriter('temp/result.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))
            # 加载蒙版，要求白色部分为生成区域，黑色为透明
            mask_path = os.path.join('.', 'models', 'mask.png')
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                raise ValueError("Mask image not found at " + mask_path)

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
        with torch.no_grad():
            pred = model(mel_batch, img_batch)
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
        
        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            # 调整预测结果到人脸区域尺寸
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            
            # 使用蒙版进行融合：先将蒙版resize到当前人脸区域的尺寸
            resized_mask = cv2.resize(mask_img, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
            # 对resize后的蒙版进行二值化（确保只有纯黑和纯白）
            _, binary_mask = cv2.threshold(resized_mask, 127, 255, cv2.THRESH_BINARY)
            binary_mask = binary_mask.astype(np.uint8)
            # 计算距离变换，得到每个像素到边缘的距离（羽化宽度设置为12像素）
            dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
            # feather_radius = 12.5
            feather_radius = 20
            alpha = np.clip(dist / feather_radius, 0, 1)
            # 对 alpha 进行高斯平滑，进一步软化边缘
            alpha = cv2.GaussianBlur(alpha, (5,5), 0)
            
            original_face = f[y1:y2, x1:x2].astype(np.float32)
            generated_face = p.astype(np.float32)
            blended_face = (alpha[..., None] * generated_face + (1 - alpha[..., None]) * original_face).astype(np.uint8)
            
            f[y1:y2, x1:x2] = blended_face
            out.write(f)
    out.release()
    command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, 'temp/result.avi', args.outfile)
    subprocess.call(command, shell=platform.system() != 'Windows')
    
#===============API接口部分=================#

# 添加资源清理函数
def cleanup_old_tasks():
    upload_folder = app.config['UPLOAD_FOLDER']
    max_task_age_hours = 24  # 设置任务的最大保存时间为24小时
    current_time = datetime.now()
    
    for task_id in listdir(upload_folder):
        task_dir = os.path.join(upload_folder, task_id)
        if not os.path.isdir(task_dir):
            continue
            
        # 获取任务文件夹的创建时间
        task_time = datetime.fromtimestamp(os.path.getctime(task_dir))
        task_age = (current_time - task_time).total_seconds() / 3600  # 转换为小时
        
        if task_age > max_task_age_hours:
            shutil.rmtree(task_dir)
            print(f"Deleted old task directory: {task_dir}")

# 启动定时清理任务
scheduler = BackgroundScheduler()
scheduler.add_job(func=cleanup_old_tasks, trigger="interval", hours=1)
scheduler.start()

@app.route('/sync_lips', methods=['POST'])
def sync_lips():
    # 检查是否包含必要的文件
    if 'video' not in request.files or 'audio' not in request.files:
        return jsonify({'status': 'error', 'message': 'Missing video or audio file'}), 400
    
    video_file = request.files['video']
    audio_file = request.files['audio']
    
    # 检查文件是否有效
    if video_file.filename == '' or audio_file.filename == '':
        return jsonify({'status': 'error', 'message': 'Invalid file'}), 400
    
    if not allowed_file(video_file.filename) or not allowed_file(audio_file.filename):
        return jsonify({'status': 'error', 'message': 'Invalid file format'}), 400
    
    # 创建唯一的任务目录
    task_id = str(uuid.uuid4())
    task_dir = os.path.join(app.config['UPLOAD_FOLDER'], task_id)
    os.makedirs(task_dir, exist_ok=True)
    
    # 保存上传的文件
    video_path = os.path.join(task_dir, secure_filename(video_file.filename))
    audio_path = os.path.join(task_dir, secure_filename(audio_file.filename))
    video_file.save(video_path)
    audio_file.save(audio_path)
    
    # 设置输出文件路径
    output_path = os.path.join(task_dir, 'result.mp4')
    
    # 启动处理线程
    def process_task():
        try:
            # 调用Wav2Lip推理代码
            wav2lip_main(argparse.Namespace(
                checkpoint_path=app.config['CHECKPOINT_PATH'],  # 使用配置中的模型路径
                face=video_path,
                audio=audio_path,
                outfile=output_path,
                static=False,
                fps=25.,
                pads=[0, 10, 0, 0],
                face_det_batch_size=4,
                wav2lip_batch_size=8,
                resize_factor=1,
                crop=[0, -1, 0, -1],
                box=[-1, -1, -1, -1],
                rotate=False,
                nosmooth=False
            ))
            
            # 更新任务状态为完成
            with open(os.path.join(task_dir, 'status.txt'), 'w') as f:
                f.write('completed')
            
        except Exception as e:
            # 更新任务状态为失败
            with open(os.path.join(task_dir, 'status.txt'), 'w') as f:
                f.write(f'failed: {str(e)}')
    
    # 启动后台线程处理任务
    threading.Thread(target=process_task).start()
    
    # 返回任务ID
    return jsonify({
        'status': 'processing',
        'task_id': task_id,
        'message': 'Task started successfully'
    }), 202

@app.route('/task_status/<task_id>', methods=['GET'])
def get_task_status(task_id):
    task_dir = os.path.join(app.config['UPLOAD_FOLDER'], task_id)
    
    # 检查任务是否存在
    if not os.path.exists(task_dir):
        return jsonify({'status': 'not_found', 'message': 'Task not found'}), 404
    
    # 检查任务状态
    status_file = os.path.join(task_dir, 'status.txt')
    if not os.path.exists(status_file):
        # 任务仍在处理中
        return jsonify({
            'status': 'processing',
            'message': 'Task is still processing'
        }), 202
    
    with open(status_file, 'r') as f:
        status = f.read()
    
    if status.startswith('completed'):
        # 任务完成，返回结果URL
        output_path = os.path.join(task_dir, 'result.mp4')
        if os.path.exists(output_path):
            return jsonify({
                'status': 'completed',
                'result_url': f'/results/{task_id}/result.mp4'
            }), 200
        else:
            return jsonify({
                'status': 'error',
                'message': 'Task completed but result not found'
            }), 500
    
    elif status.startswith('failed'):
        # 任务失败
        return jsonify({
            'status': 'failed',
            'message': status[7:]  # 返回错误信息
        }), 500

@app.route('/results/<task_id>/result.mp4', methods=['GET'])
def get_result(task_id):
    task_dir = os.path.join(app.config['UPLOAD_FOLDER'], task_id)
    output_path = os.path.join(task_dir, 'result.mp4')
    
    if not os.path.exists(output_path):
        return jsonify({'status': 'not_found', 'message': 'Result not found'}), 404
    
    # 返回视频文件
    return send_from_directory(task_dir, 'result.mp4', as_attachment=True)

# 添加错误处理
@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({
        'status': 'error',
        'message': str(e)
    }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    @app.route('/')
    def home():
        return "Wav2Lip API is running!"
    
    
    

    
    
    
# 示例使用流程
# 启动服务：
# python app.py
# 上传任务：
# curl -X POST -F "video=@input_video.mp4" -F "audio=@input_audio.wav" http://localhost:5000/sync_lips
# 查询状态：
# curl http://localhost:5000/task_status/<task_id>
# 下载结果：
# curl -O http://localhost:5000/results/<task_id>/result.mp4

# 总结来说，这段代码实现了一个功能完善的唇形同步API服务，适用于各种需要音视频同步的场景。
