# UNet
(perhaps should change to ANTsRNet)

A collection of well-known deep learning architectures ported to the R language.

## R package dependencies

* [ANTsR](https://github.com/stnava/ANTsR)

* [keras](https://github.com/rstudio/keras) (install from github: ``devtools::install_github( "rstudio/keras")``)

* abind

## Image segmentation

* UNet (2-D, 3-D)
    * [O. Ronneberger, P. Fischer, and T. Brox.  U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

## Image classification 

* AlexNet (2-D, 3-D)
    * [K. Simonyan and A. Zisserman, ImageNet Classification with Deep Convolutional Neural Networks.](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

* Vgg16/Vgg19 (2-D, 3-D)
    * [K. Simonyan and A. Zisserman, Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

* ResNet/ResNeXt (2-D, 3-D)
    * [Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.  Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

* GoogLeNet (Inception v3) (2-D)
    * [C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, A. Rabinovich, Going Deeper with Convolutions](https://arxiv.org/abs/1512.00567)
