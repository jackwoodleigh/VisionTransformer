
Thus far, this project has been aimed at improving windowed vision transformers' ability to understand greater global context, within the scope of image restoration. The Swin Transforemr has been very successful in tackling this problem by simply switching between two alternating windows, however, many models have improved on its weaknesses and achieved better results. I started this project to learn and develop an intuitive understanding of architecture design by implementing and experimenting with various methodologies from papers I found interesting. It is still on-going and I occasionally work on it in my free time. 

# Models

These models both comprise three key stages. An initial shallow feature extraction module employs convolutional layers to capture basic image features. These features are then passed to the deep feature extraction stage, which is made up of a series of residual blocks containing vision transformer layers that are responsible for mapping low-quality features to high-quality features. The vision transformer layers are the sole source of variation between these models. Finally, using a series of convolutional layers, the model reconstructs the image at an enhanced resolution. 


## Deformable Sparse Window Transformer 

![dswt](https://github.com/jackwoodleigh/VisionTransformer/blob/main/Attachments/dswtimg.png)

The idea behind DS-MSA, which builds upon concepts from DAT, introduces a deformable window attention layer where the model can learn a window configuration using offsets determined by a small convolutional layer called sparse discriminator. Additionally, the sparse discriminator also predicts scores to weight the importance of windows, and gates to determine which windows to process. The goal is to enable the model to selectively attend to a limited number of the most salient windows, thereby capping the computational cost.

A key challenge in this architecture comes from the series of discrete choices it must make. Currently, I am first selecting candidate windows based on their score using TopK with a gumbel softmax, to help with exploration, and a straight through estimator to allow the gradient to flow through. I finalize the selection of candidate windows by hard selecting based on their predicted gate value and use another straight through estimator to enable gradient flow. Their indices, with the offsets, are plugged in to torch vision’s roi_align and attention is computed on the selected regions. This function utilizes bilinear interpolation, which allows gradients to flow back to the offset predictions. Finally, scatter_add is used with a mean reduction to modify the original input with the delta difference of attention. 


|      Method      |   Urban100    | 
|------------------|----------------|
| DSWT             | 27.02 / 0.8200 | 
| SwinIR           | 27.45 / 0.8254 | 
| Swin2SR          | 27.51 / 0.8274 | 
| HAT (uses Swin)  | 27.97 / 0.8368 | 
| DRCT (uses Swin) | 28.06 / 0.8378 | 
Table showing models trained on image restoration using DIV2K+Flickr2K. Models were benchmarked using 4x image restoration with PSNR and SSIM scores. 

While DSWT does not yet match the performance of either Swin model, I am personally encouraged by the results being fairly close since this project is fairly new. It currently has 37.5% less attention calls than Swin because at a bare-minimum Swin needs two attention calls per blocks while DSWT requires an average of 1.25 attention calls in its current configuration. I am hopeful that in the future I’ll find a way to match or beat Swin's performance.

## Multi-Scale Fusion Transformer 

This model attempts to overcome windowed-attention's limited ability to capture context beyond the scope of its individual window, attention is performed at multiple spatial resolutions. With this approach, lower spatial resolutions windows take up a larger portion of the total image. This means that attention computed on a larger portion of the image can provide context of global structural detail to modify local fine-grain detail. 

### Outdated results and visuals from ~4/15/2025 of MSFT that were used for my senior project

#### Architecture (Not final version)
![arch](https://github.com/jackwoodleigh/VisionTransformer/blob/main/Attachments/Group%201.png)

#### Ground Truth / Bicubic / Ours
![sample](https://github.com/jackwoodleigh/VisionTransformer/blob/main/Attachments/comparisonzoom.png)

##### Image Quality (PSNR / SSIM)

| Method  | Set5           | Set14          | BSD100         | Urban100       | Manga109       |
|---------|----------------|----------------|----------------|----------------|----------------|
| Ours    | 30.72 / 0.8753 | 27.22 / 0.7525 | 26.59 / 0.7224 | 26.66 / 0.8252 | 29.24 / 0.9017 |


## RSMT - In Progress

This is very much still in progress with different variations that do different things. Currently, to incorporate information from neighboring windows, this model aims to modulate the output of normal windowed multi-head attention using a convolutional cross attention approach. To mitigate the significant complexity this would add, images are converted into summaries of sub-regions using convolutional layers, reducing the total size.

