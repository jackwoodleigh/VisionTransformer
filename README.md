## This project is still ongoing but here are some visual details. More to come soon.

### Abstract 

In this project, we create a novel vision transformer architecture designed to perform 4x super resolution on images. Our model comprises three key stages. Initially, a shallow feature extraction module employs convolutional layers to capture basic image features. These features are then passed to the deep feature extraction stage which utilizes vision transformer layers with window-based multi-head attention to map from low-quality features to high-quality features. To overcome windowed-attention's limited ability to capture context beyond the scope of its window, we perform attention at multiple spatial resolutions to get context of varying levels of structural detail. This enables our model to modify local fine-grain detail using the context of global features. Finally, using a series of convolutional layers, the model reconstructs the image at an enhanced resolution. Experimental results demonstrate the model's efficacy in producing high-quality reconstructions, highlighting its potential for advanced image super resolution tasks. 

### Architecture 
![arch](https://github.com/jackwoodleigh/VisionTransformer/blob/main/Group%201.png)

### Samples
![sample](https://github.com/jackwoodleigh/VisionTransformer/blob/main/comparisonzoom.png)
