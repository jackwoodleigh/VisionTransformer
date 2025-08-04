
Thus far, this project has been aimed at improving windowed vision transformers' ability to understand greater global context, within the scope of image restoration. The Swin Transforemr has been very successful in tackling this problem by simply switching between two alternating windows, however, many models have improved on its weaknesses and achieved better results. I started this project to learn and develop an intuitive understanding of architecture design by implementing and experimenting with various methodologies from papers I found interesting. It is still on-going and I occasionally work on it in my free time. 

# Models

These models both comprise three key stages. An initial shallow feature extraction module employs convolutional layers to capture basic image features. These features are then passed to the deep feature extraction stage, which is made up of a series of residual blocks containing vision transformer layers that are responsible for mapping low-quality features to high-quality features. The vision transformer layers are the sole source of variation between these models. Finally, using a series of convolutional layers, the model reconstructs the image at an enhanced resolution. 

## MSFT 

This model attempts to overcome windowed-attention's limited ability to capture context beyond the scope of its individual window, attention is performed at multiple spatial resolutions. With this approach, lower spatial resolutions windows take up a larger portion of the total image. This means that attention computed on a larger portion of the image can provide context of global structural detail to modify local fine-grain detail. 

### Outdated results and visuals from ~4/15/2025 of MSFT that were used for my senior project

#### Architecture 
![arch](https://github.com/jackwoodleigh/VisionTransformer/blob/main/Attachments/Group%201.png)

#### Ground Truth / Bicubic / Ours
![sample](https://github.com/jackwoodleigh/VisionTransformer/blob/main/Attachments/comparisonzoom.png)

##### Image Quality (PSNR / SSIM)

| Method  | Set5           | Set14          | BSD100         | Urban100       | Manga109       |
|---------|----------------|----------------|----------------|----------------|----------------|
| Ours    | 30.72 / 0.8753 | 27.22 / 0.7525 | 26.59 / 0.7224 | 26.66 / 0.8252 | 29.24 / 0.9017 |


## RSMT - In Progress

This is very much still in progress with different variations that do different things. Currently, to incorporate information from neighboring windows, this model aims to modulate the output of normal windowed multi-head attention using a convolutional cross attention approach. To mitigate the significant complexity this would add, images are converted into summaries of sub-regions using convolutional layers, reducing the total size.

