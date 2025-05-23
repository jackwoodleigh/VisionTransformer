## This project is still ongoing but here are some details as it progresses. This is relavent to ~4/15/2025

### Abstract 

In this project, we create a novel vision transformer architecture designed to perform 4x super resolution on images. Our model comprises three key stages. Initially, a shallow feature extraction module employs convolutional layers to capture basic image features. These features are then passed to the deep feature extraction stage which utilizes vision transformer layers with window-based multi-head attention to map from low-quality features to high-quality features. To overcome windowed-attention's limited ability to capture context beyond the scope of its window, we perform attention at multiple spatial resolutions. With this approach, lower spatial resolutions windows take up a larger portion of the total image. This means that attention computed on a larger portion of the image can provide context of global structural detail to modify local fine-grain detail. Finally, using a series of convolutional layers, the model reconstructs the image at an enhanced resolution. Experimental results demonstrate the model's efficacy in producing high-quality reconstructions, highlighting its potential for advanced image super resolution tasks. 

### Architecture 
![arch](https://github.com/jackwoodleigh/VisionTransformer/blob/main/Attachments/Group%201.png)

### Results so far...

#### Ground Truth / Bicubic / Ours
![sample](https://github.com/jackwoodleigh/VisionTransformer/blob/main/Attachments/comparisonzoom.png)

#### Image Quality (PSNR / SSIM)

| Method  | Set5           | Set14          | BSD100         | Urban100       | Manga109       |
|---------|----------------|----------------|----------------|----------------|----------------|
| Ours    | 30.72 / 0.8753 | 27.22 / 0.7525 | 26.59 / 0.7224 | 26.66 / 0.8252 | 29.24 / 0.9017 |
