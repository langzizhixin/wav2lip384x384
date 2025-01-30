# wav2lip_384x384 introduction
This is a project about talking faces. We use288X288 sized facial images for training, which can generate720p, 1080p, 2k Digital Humanhuman videos.
We have done the following work:
1. Add video cutting codes.
2. Add filelists to generate code.
3. Trained 1000 people, 50 hours, and over 50000 pieces of data.
4. Open sourced the checkpoint for a discriminator with 800000 steps and a val_rass of 0.29.
5. Open sourced a checkpoint for a generator with 600000 steps and a val_rass of 0.27.
6. Dear friends, you can load pre training weights for easy subsequent training.


# wav2lip-288x288 Project situation
<p align='center'>
  <b>
    <a href="https://www.bilibili.com/video/BV1zK421v7wh/?vd_source=7720ff9e037156b51374d14ee8f76b51">Video </a>
    | 
    <a href="https://github.com/langzizhixin">Project Page</a>
    |
    <a href="https://github.com/langzizhixin/wav2lip_288x288">Code</a> 
  </b>
</p> 

checkpoints for wav2lip_288x288   https://pan.baidu.com/s/1ks53RXFzN56Ksjpxspiwyw?pwd=lzzx 

## The following pictures are comparison images of the training generator training 300000 steps.
<p align='center'>  
    <img src='picture/sample-01.jpg' width='600'/>
    <img src='picture/sample-02.jpg' width='600'/>
    <img src='picture/sample-03.jpg' width='600'/>
    <img src='picture/sample-04.jpg' width='600'/>
</p>

## The following images show the loss values of training the  discriminator for 300000 steps.
<p align='center'>  
    <img src='picture/expert_loss.png' width='600'/>
</p>

## The following images show the loss values of training the generator for 300000 steps.
<p align='center'>  
    <img src='picture/syncnet_loss.png' width='600'/>
</p>

# Release Plan
For the wav2lip series, we will continue to train and release higher definition weights in the future.
The plan is as follows:
Pre training checkpoints for wav2lip_288x288 will be released in January 2025.
Pre training checkpoints for wav2lip_384x384 will be released in February 2025.
Pre training checkpoints for wav2lip_576x576 or 512x512 will be released in June 2025.

# Citing
Thank the two authors, Thank you for their wonderful work.
https://github.com/primepake/wav2lip_288x288
https://github.com/Rudrabha/Wav2Lip

# Disclaimers
This repositories made by langzizhixin from Langzizhixin Technology company 2025.1.1 , in Chengdu, China .
The above code and weights can only be used for personal/research/non-commercial purposes.
If you need a higher definition model, please contact me by email 277504483@qq.com , or add WeChat for communication: langzizhixinkeji
