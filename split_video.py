import os
from pydub import AudioSegment
from pydub.silence import detect_silence
from moviepy.editor import VideoFileClip

videos_dir = "000"
slices_dir = "data-split"
# 已有的切片
prev_slices = []
for root, dirs, files in os.walk(slices_dir):
    prev_slices.extend(dirs)

prev_slices = set(prev_slices)

for root, dirs, files in os.walk(videos_dir):
    for file in files:
        name, extension = os.path.splitext(file)
        if name in prev_slices:
            continue
        if extension != '.mp4':
            continue

        file_path = os.path.join(root, file)

        video = VideoFileClip(file_path)
        audio = video.audio
        audio.write_audiofile('tmp.wav')
        audio_segment = AudioSegment.from_wav('tmp.wav')
        silence_intervals = detect_silence(audio_segment, min_silence_len=200, silence_thresh=-45)
        os.remove("tmp.wav")
        # 保留静音部分，在静音中间设置切片点
        for i in range(len(silence_intervals)):
            sli_point = (silence_intervals[i][0] + silence_intervals[i][1]) / 2
            silence_intervals[i][0] = silence_intervals[i][1] = sli_point

        clips = []
        start_time = 0
        for silence_start, silence_end in silence_intervals:
            end_time = silence_start / 1000.0  # Convert milliseconds to seconds
            if start_time < end_time:
                clip = video.subclip(start_time, end_time)
            clips.append(clip)
            start_time = silence_end / 1000.0  # Convert milliseconds to seconds

        # Add the last segment of the video
        if start_time < video.duration:
            clips.append(video.subclip(start_time, video.duration))

        for i, clip in enumerate(clips):
            if not os.path.exists(os.path.join(slices_dir, name)):
                os.mkdir(os.path.join(slices_dir, name))
            clip.write_videofile(
                os.path.join(slices_dir, name, f'{i}.mp4'), audio_codec="aac")
