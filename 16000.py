import os
import subprocess
import shutil
import tempfile

def convert_audio_to_mono_16k(input_file):
    """
    使用ffmpeg将输入音频文件转换为16000Hz、单声道格式，并替换原文件。
    
    :param input_file: 输入音频文件路径
    """
    # 创建临时文件名
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        temp_path = temp_file.name

    try:
        # 使用ffmpeg进行转换，并输出到临时文件
        command = [
            "ffmpeg",
            "-i", input_file,
            "-ac", "1",  # 设置为单声道
            "-ar", "16000",  # 设置采样率为16000Hz
            "-y",  # 覆盖输出文件（如果存在）
            temp_path
        ]
        
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 替换原文件
        shutil.move(temp_path, input_file)
        print(f"Converted and replaced {input_file}")
    
    except subprocess.CalledProcessError as e:
        print(f"Error processing {input_file}: {e.stderr.decode()}")
    finally:
        # 确保在发生错误时删除临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)

def process_directory(root_dir):
    """
    遍历指定目录及其子目录中的所有音频文件，并将它们转换为16000Hz、单声道格式。
    
    :param root_dir: 根目录路径
    """
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                input_file = os.path.join(subdir, file)
                
                print(f"Processing {input_file}...")
                
                convert_audio_to_mono_16k(input_file)

# 设置根目录路径
root_directory = "preprocessed_data"

# 开始处理
process_directory(root_directory)