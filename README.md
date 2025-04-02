## 🔥 wav2lip384x384 introduction
This is a project about talking faces. We use384X384 sized facial images for training, which can generate720p, 1080p, 2k ,4k Digital Humanhuman videos.
We have done the following work:
1. Add video cutting codes.
2. Add filelists to generate code.
3. Trained 1000 people, 50 hours, and over 50000 pieces of data.
4. Open sourced the checkpoint for a discriminator with 150000, 700000, 1000000 steps and a val_loss of 0.36, 0.33, 0.28.
5. open-source  the checkpoint for generator with 300000, 500000, and 800000 steps, with val_loss values of 0.35, 0.32, and 0.29 , performs very well and is recommended for use. Of course, it can also be loaded for further training.
6. Dear friends,we did not release the best generator checkpoint, However, generators with over 500000 steps have surpassed all open source projects on the market in terms of direct inference performance, and have reached a basic commercial level.
7. Dear friends,Although we did not release the best generator checkpoint, but we released the best discriminator checkpoint, you need load pre training weights for easy subsequent training, many people have loaded our color_checkpoints and final_checkpionts for training, and achieved good results.Especially when solving profile and occlusion problems, it is only necessary to load the relevant dataset and continue training.
8. Due to the wav2lip high-definition algorithm series, it cannot achieve high fidelity of faces and teeth, and the training difficulty is relatively high, which cannot adapt well to current commercial needs. So we have changed the algorithm for commercial digital humans and adopted new algorithms such as diffusion.
9. Friends who want to train the wav2lip high-definition series, please think carefully before taking action.

## 🏗️ wav2lip-384x384 Project situation
<p align='center'>
  <b>
    <a href="https://www.bilibili.com/video/BV1WsFVePEuz/?spm_id_from=333.1387.upload.video_card.click">Video </a>
    | 
    <a href="https://github.com/langzizhixin">Project Page</a>
    |
    <a href="https://github.com/langzizhixin/wav2lip384x384">Code</a> 
  </b>
</p> 

checkpoints for wav2lip384x384   https://pan.baidu.com/s/1NiSEdrlRVZM_6SD4Igdtlg?pwd=lzzx 

## 📊 The following pictures are comparison images of the training generator training 500000 steps.
<p align='center'>  
    <img src='picture/11.jpg' width='1200'/>
</p>

## 🎬 Demo
<video width="640" height="480" controls>
  <source src="https://raw.githubusercontent.com/langzizhixin/wav2lip384x384/main/picture/input.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>



## 🎬 Demo
<video width="640" height="480" controls>
  <source src="https://github.com/langzizhixin/wav2lip384x384/blob/main/picture/input.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>


<table class="center">
  <tr style="font-weight: bolder;text-align:center;">
        <td width="50%"><b>Original video</b></td>
        <td width="50%"><b>Lip-synced video</b></td>
  </tr>
  <tr>
    <td>
      <video src=https://github.com/langzizhixin/wav2lip384x384/blob/main/picture/3d%E8%B6%85%E5%86%99%E5%AE%9E.mp4 controls preload></video>
    </td>
    <td>
      <video src='picture/output-01.mp4' controls preload></video>
    </td>
  </tr>
  <tr>
    <td>
      <video src=https://github.com/user-attachments/assets/32c830a9-4d7d-4044-9b33-b184d8e11010 controls preload></video>
    </td>
    <td>
      <video src=https://github.com/user-attachments/assets/84e4fe9d-b108-44a4-8712-13a012348145 controls preload></video>
    </td>
  </tr>
  <tr>
    <td>
      <video src=https://github.com/user-attachments/assets/7510a448-255a-44ee-b093-a1b98bd3961d controls preload></video>
    </td>
    <td>
      <video src=https://github.com/user-attachments/assets/6150c453-c559-4ae0-bb00-c565f135ff41 controls preload></video>
    </td>
  </tr>
  <tr>
    <td width=300px>
      <video src=https://github.com/user-attachments/assets/0f7f9845-68b2-4165-bd08-c7bbe01a0e52 controls preload></video>
    </td>
    <td width=300px>
      <video src=https://github.com/user-attachments/assets/c34fe89d-0c09-4de3-8601-3d01229a69e3 controls preload></video>
    </td>
  </tr>
  <tr>
    <td>
      <video src=https://github.com/user-attachments/assets/7ce04d50-d39f-4154-932a-ec3a590a8f64 controls preload></video>
    </td>
    <td>
      <video src=https://github.com/user-attachments/assets/70bde520-42fa-4a0e-b66c-d3040ae5e065 controls preload></video>
    </td>
  </tr>
</table>

## 📑 Open-source Plan
For the wav2lip series, we will continue to train and release higher definition weights in the future.
The plan is as follows:
Pre training checkpoints for wav2lip_288x288 will be released in January 2025.
Pre training checkpoints for wav2lip_384x384 will be released in February 2025.
Pre training checkpoints for wav2lip_576x576 or 512x512 will be released after June 2025.
- [x] color_checkpoints  
- [x] final_checkpionts
- [x] Dataset processing pipeline
- [x] Training method
- [ ] Real time Inference 
- [ ] Advanced Inference
- [ ] Higher definition commercial checkpoints



## 🙏  Citing
Thank you to the other three authors, Thank you for their wonderful work.

https://github.com/primepake/wav2lip_288x288

https://github.com/nghiakvnvsd/wav2lip384

https://github.com/Rudrabha/Wav2Lip

## 📖 Disclaimers
This repositories made by langzizhixin from Langzizhixin Technology company 2025.1.30 , in Chengdu, China .
The above code and weights can only be used for personal/research/non-commercial purposes.
If you need a higher definition model, please contact me by email 277504483@qq.com , or add ours WeChat for communication: langzizhixinkeji
