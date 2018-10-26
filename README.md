# ANTsRNetExamples

A collection of examples and models for the [ANTsRNet](https://github.com/ANTsX/ANTsRNet) package.

* Image voxelwise segmentation/regression
* Image classification/regression
* Object detection
* Image super-resolution
    
---------------------------------

# Misc topics

* Optimizers
* [Blog:  Important papers](https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html)
* [Blog:  Intuitive explanation of convnets](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/)
* [Deep learning book](http://www.deeplearningbook.org)
* [Keras functional API](https://cran.r-project.org/web/packages/keras/vignettes/functional_api.html)
* Important Keras [FAQ](https://keras.rstudio.com/articles/faq.html)
* Custom keras layers in R
    * [Link 1](https://keras.rstudio.com/articles/custom_layers.html)
    * [Link 2](https://cran.rstudio.com/web/packages/keras/vignettes/about_keras_layers.html)
    * [Link 3](https://cran.rstudio.com/web/packages/keras/vignettes/custom_layers.html)

# To do:

* ResNet and AlexNet use lambda layers so those models aren't writeable to file (h5 format).  So we need to redo to rewrite to json or something else.  At least I think that's the problem. 
* Need to go through and make sure that the 'tf' vs. 'th' ordering is accounted for.  Currently, tensorflow is assumed.  Should work with theano but need to check this.  Actually, given that Theano is [no longer in active development](https://groups.google.com/forum/#!topic/theano-users/7Poq8BZutbY), perhaps we should just stick with a tensorflow backend.

****************
****************

# My GPU set-up

## Hardware

* Computer 
    * iMac (27-inch, Mid 2011)
    * Processor 3.4 GHz Intel Core i7
    * Memory 16 GB 1333 MHz DDR3 
    * macOS High Sierra (Version 10.13.2)
* GPU
    * [NVIDIA Titan Xp](https://www.nvidia.com/en-us/titan/titan-xp/)
    * [Akitio Node - Thunderbolt3 eGPU](https://www.akitio.com/expansion/node)
    * [Thunderbolt 3 <--> Thunderbolt 2 adapter](https://www.apple.com/shop/product/MMEL2AM/A/thunderbolt-3-usb-c-to-thunderbolt-2-adapter)
    * [Thunderbolt 2 cable](https://www.apple.com/shop/product/MD862LL/A/apple-thunderbolt-cable-2-m)

## Software

* Tensorflow-gpu
* Keras in R
* [NVIDIA CUDA toolkit 9.1](https://developer.nvidia.com/cuda-downloads?target_os=MacOSX&target_arch=x86_64&target_version=1012)
* [NVIDIA CUDA Deep Neural Network library (cuDNN) 7.0](https://www.developer.nvidia.com/cudnn)
* Python 3.6

## Set-up

(see note in Misc. about when to plug in/turn on eGPU)

1. [Put together Titan XP and Aikito node](https://becominghuman.ai/deep-learning-gaming-build-with-nvidia-titan-xp-and-macbook-pro-with-thunderbolt2-5ceee7167f8b)
2. [Install web drivers and GPU support](https://egpu.io/forums/mac-setup/wip-nvidia-egpu-support-for-high-sierra/)
3. Install NVIDIA toolkit and cuDNN
4. Re-install web drivers and GPU support
5. [Install tensorflow-gpu](https://medium.com/@fabmilo/how-to-compile-tensorflow-with-cuda-support-on-osx-fd27108e27e1)    
6. [Install keras with tensorflow-gpu](https://keras.rstudio.com)

__Update (April 11, 2018):__ The recent MacOSx update (10.13.4) broke the eGPU compatiblity as explained [here](https://egpu.io/forums/mac-setup/script-enable-egpu-on-tb1-2-macs-on-macos-10-13-4/).  

## Misc. notes

* I originally set-up the hardware followed by the drivers (steps 1 and 2) but the tensorflow installation caused some problems.  I believe they were from ``csrutil enable --without kext`` instead of ``csrutil disable`` in step 2 so I ended up using the latter.
* As described in the [comments](https://gist.github.com/smitshilu/53cf9ff0fd6cdb64cca69a7e2827ed0f), I had to change the following files:
    * tensorflow/third_party/gpus/cuda/BUILD.tpl (comment out line 113 ``linkopts = ["-lgomp"],``)
    * tensorflow/core/kernels/depthwise_conv_op_gpu.cu.cc (remove all instances of ``align(sizeof(T))``)
    * tensorflow/core/kernels/split_lib_gpu.cu.cc (remove all instances of ``align(sizeof(T))``)
    * tensorflow/core/kernels/concat_lib_gpu.impl.cu.cc (remove all instances of ``align(sizeof(T))``)
* Since I ended up re-installing the NVIDIA drivers, I think I should have performed Step 3 before Step 2 in the Set-up above.  
* I had to revert back to older Xcode and command line tools (8.3.2) and then switch back.  


* Time differences on [MNIST example](https://github.com/ntustison/ANTsRNet/blob/master/Examples/AlexNetExample/mnist.R)
    * tensorflow-cpu on Mac Pro (Late 2013):  ~2100 seconds / epoch
    * tensorflow-gpu (the described set-up):  ~97 seconds / epoch
* Time differences on [U-net example](https://github.com/ntustison/ANTsRNet/tree/master/Examples/UnetExample)
    * tensorflow-cpu on Mac Pro (Late 2013):  ~56 seconds / epoch
    * tensorflow-gpu (the described set-up):  ~2 seconds / epoch

* During a run a kernel panic resulted in the computer shutting down.  When it came back on, the GPU was no longer recognized but was listed as a "NVIDIA chip" in the "About this mac" --> "System Report" --> "Graphics/Displays".  Reinstalling the web driver and eGPU support didn't bring it back but then I read where I needed to 
    1. unplug the eGPU
    2. Boot into OSX
    3. Login
    4. Plug in the eGPU
    5. Logout
    6. Log back in

****************
****************

# My laptop set-up 

## Hardware

* MacBook Pro (13-inch, 2016, Four Thunderbolt 3 Ports)
* Blackmagic eGPU (AMD Radeon Pro 580)

## Software

* [plaidml](https://github.com/plaidml/plaidml)
* keras in R
* anaconda3 

## Misc. notes

* Installed plaidml according to the [instructions](https://github.com/plaidml/plaidml/blob/master/docs/installing.md).
* During the ``plaidml-setup`` call, I selected something like ``2 : metal_amd_radeon_pro_580.0``.
* Somewhere I read I should install anaconda3
* Switched the python library in ``~/.Rprofile``
* Uninstalled keras in R (``> remove.package( "keras" )``) as that was tied to the other version of python (``/Library/Frameworks/Python.framework/Versions/3.6/bin/python3.6``).
* Reinstalled keras which added everything to the anaconda3 python directory
    * ``> install.packages( "keras" )``
    * ``> install_keras()``
* Despite this, I could not get keras in R to switch to the "plaidml" backend.
    * ``> library( keras )``
    * ``> use_backend( "plaidml" )``
    * ``> K <- backend()``
    * ``> K$backend()``
    * ``[1] "tensorflow"``
* I even tried to change the backend in ``~/.keras/keras.json`` but no luck.    
* I also noticed that ``install_keras()`` activates a conda environment (``r-tensorflow``) so I reinstalled plaidml in that conda environment but still no luck getting keras in R to switch backends.
* However, I could get the python plaidml [example](https://github.com/plaidml/plaidml#hello-vgg) to work.  I could open up ``Activity Monitor --> Window --> GPU History`` and see the workload on the Blackmagic eGPU.
* Finally, after doing that, I noticed the line in the python example ``os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"``.  I finally got the plaidml backend to work by adding that environment variable, i.e., ``$ export KERAS_BACKEND="plaidml.keras.backend"``.  After that, I started up and was able to run this [example](https://rpubs.com/siero5335/399690) and see the workload on the eGPU:

```
$ R

R version 3.5.1 (2018-07-02) -- "Feather Spray"
Copyright (C) 2018 The R Foundation for Statistical Computing
Platform: x86_64-apple-darwin15.6.0 (64-bit)

R is free software and comes with ABSOLUTELY NO WARRANTY.
You are welcome to redistribute it under certain conditions.
Type 'license()' or 'licence()' for distribution details.

  Natural language support but running in an English locale

R is a collaborative project with many contributors.
Type 'contributors()' for more information and
'citation()' on how to cite R or R packages in publications.

Type 'demo()' for some demos, 'help()' for on-line help, or
'help.start()' for an HTML browser interface to help.
Type 'q()' to quit R.


*** Successfully loaded .Rprofile ***

##------ [/Users/ntustison/Pkg] Thu Oct 25 20:56:11 2018 ------##
> library( keras )
Using plaidml.keras.backend backend.
> use_backend(backend = "plaidml")
> K <- backend()
> K$backend()
[1] "plaidml.keras.backend"
> batch_size <- 128
> num_classes <- 10
> epochs <- 5
> img_rows <- 28
> img_cols <- 28
> 
> 
> mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y
> x_train <- mnist$train$x
> y_train <- mnist$train$y
> x_test <- mnist$test$x
> y_test <- mnist$test$y
> x_train <- array_reshape(x_train, c(nrow(x_train), img_rows, img_cols, 1))
> x_test <- array_reshape(x_test, c(nrow(x_test), img_rows, img_cols, 1))
> input_shape <- c(img_rows, img_cols, 1)
> 
> x_train <- x_train / 255
> x_test <- x_test / 255
> 
> y_train <- to_categorical(y_train, 10)
> y_test <- to_categorical(y_test, 10)
> 
> model <- keras_model_sequential() %>%
+   layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
+                 input_shape = input_shape) %>% 
+   layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
+   layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
+   layer_dropout(rate = 0.25) %>% 
+   layer_flatten() %>% 
+   layer_dense(units = 128, activation = 'relu') %>% 
+   layer_dropout(rate = 0.5) %>% 
+   layer_dense(units = num_classes, activation = 'softmax')
INFO:plaidml:Opening device "metal_amd_radeon_pro_580.0"
> 
> summary(model)
________________________________________________________________________________
Layer (type)                        Output Shape                    Param #     
================================================================================
conv2d_1 (Conv2D)                   (None, 26, 26, 32)              320         
________________________________________________________________________________
conv2d_2 (Conv2D)                   (None, 24, 24, 64)              18496       
________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)      (None, 12, 12, 64)              0           
________________________________________________________________________________
dropout_1 (Dropout)                 (None, 12, 12, 64)              0           
________________________________________________________________________________
flatten_1 (Flatten)                 (None, 9216)                    0           
________________________________________________________________________________
dense_1 (Dense)                     (None, 128)                     1179776     
________________________________________________________________________________
dropout_2 (Dropout)                 (None, 128)                     0           
________________________________________________________________________________
dense_2 (Dense)                     (None, 10)                      1290        
================================================================================
Total params: 1,199,882
Trainable params: 1,199,882
Non-trainable params: 0
________________________________________________________________________________
> 
> model %>% compile(
+   loss = loss_categorical_crossentropy,
+   optimizer = optimizer_adadelta(),
+   metrics = c('accuracy')
+ )
> model %>% fit(
+   x_train, y_train,
+   batch_size = batch_size,
+   epochs = epochs,
+   validation_split = 0.2
+ )
Train on 48000 samples, validate on 12000 samples
Epoch 1/5
INFO:plaidml:Analyzing Ops: 85 of 285 operations complete
48000/48000 [==============================] - 18s 372us/step - loss: 0.3130 - acc: 0.9042 - val_loss: 0.0702 - val_acc: 0.9803
Epoch 2/5
48000/48000 [==============================] - 10s 200us/step - loss: 0.0982 - acc: 0.9705 - val_loss: 0.0543 - val_acc: 0.9846
Epoch 3/5
48000/48000 [==============================] - 9s 188us/step - loss: 0.0721 - acc: 0.9787 - val_loss: 0.0490 - val_acc: 0.9867
Epoch 4/5
48000/48000 [==============================] - 9s 188us/step - loss: 0.0613 - acc: 0.9814 - val_loss: 0.0410 - val_acc: 0.9882
Epoch 5/5
48000/48000 [==============================] - 9s 187us/step - loss: 0.0510 - acc: 0.9851 - val_loss: 0.0418 - val_acc: 0.9882
```
    
