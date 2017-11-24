#' 2-D implementation of the AlexNet deep learning architecture.
#'
#' Creates a keras model of the AlexNet deep learning architecture for image 
#' recognition based on the paper
#' 
#' A. Krizhevsky, and I. Sutskever, and G. Hinton. ImageNet Classification 
#'   with Deep Convolutional Neural Networks.
#' 
#' available here:
#' 
#'         http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
#'
#' This particular implementation was influenced by the following python 
#' implementation: 
#' 
#'         https://github.com/duggalrahul/AlexNet-Experiments-Keras/     
#'         https://github.com/lunardog/convnets-keras/
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).  The batch size
#' (i.e., number of training images) is not specified a priori. 
#' @param numberOfClassificationLabels Number of segmentation labels.  
#'
#' @return a AlexNet keras model to be used with subsequent fitting
#' @author Tustison NJ
#' @examples
#' # Simple examples, must run successfully and quickly. These will be tested.
#' \dontrun{ 
#' 
#'  library( ANTsR )
#'
#'  imageIDs <- c( "r16", "r27", "r30", "r62", "r64", "r85" )
#'
#'  # Perform simple 3-tissue segmentation.  For convenience we are going 
#   # to use kmeans segmentation to define the "ground-truth" segmentations.
#'  
#'  segmentationLabels <- c( 1, 2, 3 )
#'  
#'  images <- list()
#'  kmeansSegs <- list()
#'
#'  trainingImageArrays <- list()
#'  trainingMaskArrays <- list()
#'
#'  for( i in 1:length( imageIDs ) )
#'    {
#'    cat( "Processing image", imageIDs[i], "\n" )
#'    images[[i]] <- antsImageRead( getANTsRData( imageIDs[i] ) )
#'    mask <- getMask( images[[i]] )
#'    kmeansSegs[[i]] <- kmeansSegmentation( images[[i]], 
#'      length( segmentationLabels ), mask, mrf = 0.0 )$segmentation
#' 
#'    trainingImageArrays[[i]] <- as.array( images[[i]] )
#'    trainingMaskArrays[[i]] <- as.array( mask )
#'    }
#'  
#'  # reshape the training data to the format expected by keras
#'  
#'  trainingLabelData <- abind( trainingMaskArrays, along = 3 )  
#'  trainingLabelData <- aperm( trainingLabelData, c( 3, 1, 2 ) )
#'
#'  trainingData <- abind( trainingImageArrays, along = 3 )   
#'  trainingData <- aperm( trainingData, c( 3, 1, 2 ) )
#'  
#'  # Perform an easy normalization which is important for U-net. 
#'  # Other normalization methods might further improve results.
#'  
#'  trainingData <- ( trainingData - mean( trainingData ) ) / sd( trainingData )
#'
#'  X_train <- array( trainingData, dim = c( dim( trainingData ), 
#'    numberOfClassificationLabels = length( segmentationLabels ) ) )
#'  Y_train <- array( trainingLabelData, dim = c( dim( trainingData ), 
#'    numberOfClassificationLabels = length( segmentationLabels ) ) )
#'  
#'  # Create the model
#'  
#'  outputs <- createAlexNetModel2D( dim( trainingImageArrays[[1]] ), 
#'    numberOfClassificationLabels = numberOfLabels, layers = 1:4 )
#'  
#'  # Fit the model
#'  
#'  track <- outputs %>% fit( X_train, Y_train, 
#'                 epochs = 100, batch_size = 32, verbose = 1, shuffle = TRUE,
#'                 callbacks = list( 
#'                   callback_model_checkpoint( paste0( baseDirectory, "weights.h5" ), 
#'                      monitor = 'val_loss', save_best_only = TRUE ),
#'                 #  callback_early_stopping( patience = 2, monitor = 'loss' ),
#'                   callback_reduce_lr_on_plateau( monitor = "val_loss", factor = 0.1 )
#'                 ), 
#'                 validation_split = 0.2 )
#'
#'  # Save the model and/or save the model weights
#'
#'  save_model_hdf5( unetModel, filepath = 'unetModel.h5' )
#'  save_model_weights_hdf5( unetModel, filepath = 'unetModelWeights.h5' ) )
#' }

createAlexNetModel2D <- function( inputImageSize, 
                                  numberOfClassificationLabels = 1000,
                                  denseUnits = 4096,
                                  dropoutRate = 0.0
                                )
{

  if ( ! usePkg( "keras" ) )
    {
    stop( "Please install the keras package." )
    }

  splitTensor2D <- function( axis = 4, ratioSplit = 1, idSplit = 1  )
    {
    f <- function( X )
      {
      Xdims <- K$int_shape( X )
      div <- as.integer( Xdims[[axis]] / ratioSplit )
      axisSplit <- ( ( idSplit - 1 ) * div + 1 ):( idSplit * div )  

      if( axis == 1 ) 
        {
        output <- X[axisSplit,,,]
        } else if( axis == 2 ) {
        output <- X[, axisSplit,,]
        } else if( axis == 3 ) {
        output <- X[,, axisSplit,]
        } else if( axis == 4 ) {
        output <- X[,,, axisSplit]
        } else {
        stop( "Wrong axis specification." )  
        }
      return( output )
      }

    return( layer_lambda( f = f ) )
    }  

  crossChannelNormalization2D <- function( alpha = 1e-4, k = 2, beta = 0.75, n = 5L )
    {
    K <- keras::backend()       
 
    normalizeTensor2D <- function( X )
      {
      #  Theano:  [batchSize, channelSize, widthSize, heightSize]
      #  tensorflow:  [batchSize, widthSize, heightSize, channelSize]

      if( K$backend() == 'tensorflow' )
        {
        Xshape <- X$get_shape()  
        } else {
        Xshape <- X$shape()  
        }
      X2 <- K$square( X )

      half <- as.integer( n / 2 )

      extraChannels <- K$spatial_2d_padding( 
        K$permute_dimensions( X2, c( 1L, 2L, 3L, 0L ) ), 
        padding = list( c( 0L, 0L ), c( half, half ) ) )
      extraChannels <- K$permute_dimensions( extraChannels, c( 3L, 0L, 1L, 2L ) )  
      scale <- k

      Xdims <- K$int_shape( X )
      ch <- as.integer( Xdims[[length( Xdims )]] )
      for( i in 1:n )
        {
        scale <- scale + alpha * extraChannels[,,, i:( i + ch - 1 )]  
        }
      scale <- K$pow( scale, beta )

      return( X / scale )
      }

    return( layer_lambda( f = normalizeTensor2D ) )  
    }
    
  K <- keras::backend()
  inputs <- layer_input( shape = inputImageSize )

  # Conv1
  outputs <- inputs %>% layer_conv_2d( filters = 96, 
    kernel_size = c( 11, 11 ), strides = c( 4, 4 ), activation = 'relu' )

  # Conv2
  outputs <- outputs %>% layer_max_pooling_2d( pool_size = c( 3, 3 ), 
    strides = c( 2, 2 ) )
  normalizationLayer <- crossChannelNormalization2D()
  outputs <- outputs %>% normalizationLayer

  outputs <- outputs %>% layer_zero_padding_2d( padding = c( 2, 2 ) )

  convolutionLayer <- outputs %>% layer_conv_2d( filters = 128, 
    kernel_size = c( 5, 5 ), padding = 'same' )
  lambdaLayers <- list( convolutionLayer )
  for( i in 1:2 )
    {
    splitLayer <- splitTensor2D( axis = 4, ratioSplit = 2, idSplit = i )
    lambdaLayers <- lappend( lambdaLayers, outputs %>% splitLayer )
    }
  outputs <- layer_concatenate( lambdaLayers )

  # Conv3
  outputs <- outputs %>% layer_max_pooling_2d( pool_size = c( 3, 3 ), 
    strides = c( 2, 2 ) )
  normalizationLayer <- crossChannelNormalization2D()
  outputs <- outputs %>% normalizationLayer

  outputs <- outputs %>% layer_zero_padding_2d( padding = c( 2, 2 ) )
  outputs <- outputs %>% layer_conv_2d( filters = 384, 
    kernel_size = c( 3, 3 ), padding = 'same' )

  # Conv4
  outputs <- outputs %>% layer_zero_padding_2d( padding = c( 2, 2 ) )
  convolutionLayer <- outputs %>% layer_conv_2d( filters = 192, 
    kernel_size = c( 3, 3 ), padding = 'same' )
  lambdaLayers <- list( convolutionLayer )
  for( i in 1:2 )
    {
    splitLayer <- splitTensor2D( axis = 4, ratioSplit = 2, idSplit = i )
    lambdaLayers <- lappend( lambdaLayers, outputs %>% splitLayer )
    }
  outputs <- layer_concatenate( lambdaLayers )

  # Conv5
  outputs <- outputs %>% layer_zero_padding_2d( padding = c( 2, 2 ) )
  normalizationLayer <- crossChannelNormalization2D()
  outputs <- outputs %>% normalizationLayer

  convolutionLayer <- outputs %>% layer_conv_2d( filters = 128, 
    kernel_size = c( 3, 3 ), padding = 'same' )
  lambdaLayers <- list( convolutionLayer )
  for( i in 1:2 )
    {
    splitLayer <- splitTensor2D( axis = 4, ratioSplit = 2, idSplit = i )
    lambdaLayers <- lappend( lambdaLayers, outputs %>% splitLayer )
    }
  outputs <- layer_concatenate( lambdaLayers )

  outputs <- outputs %>% layer_max_pooling_2d( pool_size = c( 3, 3 ), strides = c(2, 2 ) )
  outputs <- outputs %>% layer_flatten()
  outputs <- outputs %>% layer_dense( units = denseUnits, activation = 'relu' )
  if( dropoutRate > 0.0 )
    {
    outputs <- outputs %>% layer_dropout( rate = dropoutRate )
    }
  outputs <- outputs %>% layer_dense( units = denseUnits, activation = 'relu' )
  if( dropoutRate > 0.0 )
    {
    outputs <- outputs %>% layer_dropout( rate = dropoutRate )
    }
  outputs <- outputs %>% layer_dense( units = numberOfClassificationLabels, activation = 'softmax' )

  alexNetModel <- keras_model( inputs = inputs, outputs = outputs )

  return( alexNetModel )
}

#' 3-D implementation of the AlexNet deep learning architecture.
#'
#' Creates a keras model of the AlexNet deep learning architecture for image 
#' recognition based on the paper
#' 
#' A. Krizhevsky, and I. Sutskever, and G. Hinton. ImageNet Classification 
#'   with Deep Convolutional Neural Networks.
#' 
#' available here:
#' 
#'         http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
#'
#' This particular implementation was influenced by the following python 
#' implementation: 
#' 
#'         https://github.com/duggalrahul/AlexNet-Experiments-Keras/     
#'         https://github.com/lunardog/convnets-keras/
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).  The batch size
#' (i.e., number of training images) is not specified a priori. 
#' @param numberOfClassificationLabels Number of segmentation labels.  
#'
#' @return a AlexNet keras model to be used with subsequent fitting
#' @author Tustison NJ
#' @examples
#' # Simple examples, must run successfully and quickly. These will be tested.
#' \dontrun{ 
#' 
#'  library( ANTsR )
#'
#'  imageIDs <- c( "r16", "r27", "r30", "r62", "r64", "r85" )
#'
#'  # Perform simple 3-tissue segmentation.  For convenience we are going 
#   # to use kmeans segmentation to define the "ground-truth" segmentations.
#'  
#'  segmentationLabels <- c( 1, 2, 3 )
#'  
#'  images <- list()
#'  kmeansSegs <- list()
#'
#'  trainingImageArrays <- list()
#'  trainingMaskArrays <- list()
#'
#'  for( i in 1:length( imageIDs ) )
#'    {
#'    cat( "Processing image", imageIDs[i], "\n" )
#'    images[[i]] <- antsImageRead( getANTsRData( imageIDs[i] ) )
#'    mask <- getMask( images[[i]] )
#'    kmeansSegs[[i]] <- kmeansSegmentation( images[[i]], 
#'      length( segmentationLabels ), mask, mrf = 0.0 )$segmentation
#' 
#'    trainingImageArrays[[i]] <- as.array( images[[i]] )
#'    trainingMaskArrays[[i]] <- as.array( mask )
#'    }
#'  
#'  # reshape the training data to the format expected by keras
#'  
#'  trainingLabelData <- abind( trainingMaskArrays, along = 3 )  
#'  trainingLabelData <- aperm( trainingLabelData, c( 3, 1, 2 ) )
#'
#'  trainingData <- abind( trainingImageArrays, along = 3 )   
#'  trainingData <- aperm( trainingData, c( 3, 1, 2 ) )
#'  
#'  # Perform an easy normalization which is important for U-net. 
#'  # Other normalization methods might further improve results.
#'  
#'  trainingData <- ( trainingData - mean( trainingData ) ) / sd( trainingData )
#'
#'  X_train <- array( trainingData, dim = c( dim( trainingData ), 
#'    numberOfClassificationLabels = length( segmentationLabels ) ) )
#'  Y_train <- array( trainingLabelData, dim = c( dim( trainingData ), 
#'    numberOfClassificationLabels = length( segmentationLabels ) ) )
#'  
#'  # Create the model
#'  
#'  outputs <- createAlexNetModel2D( dim( trainingImageArrays[[1]] ), 
#'    numberOfClassificationLabels = numberOfLabels, layers = 1:4 )
#'  
#'  # Fit the model
#'  
#'  track <- outputs %>% fit( X_train, Y_train, 
#'                 epochs = 100, batch_size = 32, verbose = 1, shuffle = TRUE,
#'                 callbacks = list( 
#'                   callback_model_checkpoint( paste0( baseDirectory, "weights.h5" ), 
#'                      monitor = 'val_loss', save_best_only = TRUE ),
#'                 #  callback_early_stopping( patience = 2, monitor = 'loss' ),
#'                   callback_reduce_lr_on_plateau( monitor = "val_loss", factor = 0.1 )
#'                 ), 
#'                 validation_split = 0.2 )
#'
#'  # Save the model and/or save the model weights
#'
#'  save_model_hdf5( unetModel, filepath = 'unetModel.h5' )
#'  save_model_weights_hdf5( unetModel, filepath = 'unetModelWeights.h5' ) )
#' }

createAlexNetModel3D <- function( inputImageSize, 
                                  numberOfClassificationLabels = 1000,
                                  denseUnits = 4096,
                                  dropoutRate = 0.0
                                )
{

  if ( ! usePkg( "keras" ) )
    {
    stop( "Please install the keras package." )
    }

  splitTensor3D <- function( axis = 4, ratioSplit = 1, idSplit = 1  )
    {
    f <- function( X )
      {
      Xdims <- K$int_shape( X )
      div <- as.integer( Xdims[[axis]] / ratioSplit )
      axisSplit <- ( ( idSplit - 1 ) * div + 1 ):( idSplit * div )  

      if( axis == 1 ) 
        {
        output <- X[axisSplit,,,,]
        } else if( axis == 2 ) {
        output <- X[, axisSplit,,,]
        } else if( axis == 3 ) {
        output <- X[,, axisSplit,,,]
        } else if( axis == 4 ) {
        output <- X[,,, axisSplit,]
        } else if( axis == 5 ) {
        output <- X[,,,, axisSplit]
        } else {
        stop( "Wrong axis specification." )  
        }
      return( output )
      }

    return( layer_lambda( f = f ) )
    }  

  crossChannelNormalization3D <- function( alpha = 1e-4, k = 2, beta = 0.75, n = 5L )
    {
    K <- keras::backend()       
 
    normalizeTensor3D <- function( X )
      {
      #  Theano:  [batchSize, channelSize, widthSize, heightSize, depthSize]
      #  tensorflow:  [batchSize, widthSize, heightSize, depthSize, channelSize]

      if( K$backend() == 'tensorflow' )
        {
        Xshape <- X$get_shape()  
        } else {
        Xshape <- X$shape()  
        }
      X2 <- K$square( X )

      half <- as.integer( n / 2 )

      extraChannels <- K$spatial_3d_padding( 
        K$permute_dimensions( X2, c( 1L, 2L, 3L, 4L, 0L ) ), 
        padding = list( c( 0L, 0L ), c( 0L, 0L ), c( half, half ) ) )
      extraChannels <- K$permute_dimensions( extraChannels, c( 4L, 0L, 1L, 2L, 3L ) )  
      scale <- k

      Xdims <- K$int_shape( X )
      ch <- as.integer( Xdims[[length( Xdims )]] )
      for( i in 1:n )
        {
        scale <- scale + alpha * extraChannels[,,,, i:( i + ch - 1 )]  
        }
      scale <- K$pow( scale, beta )

      return( X / scale )
      }

    return( layer_lambda( f = normalizeTensor3D ) )  
    }
    
  K <- keras::backend()
  inputs <- layer_input( shape = inputImageSize )

  # Conv1
  outputs <- inputs %>% layer_conv_3d( filters = 96, 
    kernel_size = c( 11, 11, 11 ), strides = c( 4, 4, 4 ), activation = 'relu' )

  # Conv2
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = c( 3, 3, 3 ), 
    strides = c( 2, 2, 2 ) )
  normalizationLayer <- crossChannelNormalization3D()
  outputs <- outputs %>% normalizationLayer

  outputs <- outputs %>% layer_zero_padding_3d( padding = c( 2, 2, 2 ) )

  convolutionLayer <- outputs %>% layer_conv_3d( filters = 128, 
    kernel_size = c( 5, 5, 5 ), padding = 'same' )
  lambdaLayers <- list( convolutionLayer )
  for( i in 1:2 )
    {
    splitLayer <- splitTensor2D( axis = 5, ratioSplit = 2, idSplit = i )
    lambdaLayers <- lappend( lambdaLayers, outputs %>% splitLayer )
    }
  outputs <- layer_concatenate( lambdaLayers )

  # Conv3
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = c( 3, 3, 3 ), 
    strides = c( 2, 2, 2 ) )
  normalizationLayer <- crossChannelNormalization3D()
  outputs <- outputs %>% normalizationLayer

  outputs <- outputs %>% layer_zero_padding_3d( padding = c( 2, 2, 2 ) )
  outputs <- outputs %>% layer_conv_3d( filters = 384, 
    kernel_size = c( 3, 3, 3 ), padding = 'same' )

  # Conv4
  outputs <- outputs %>% layer_zero_padding_3d( padding = c( 2, 2, 2 ) )
  convolutionLayer <- outputs %>% layer_conv_3d( filters = 192, 
    kernel_size = c( 3, 3, 3 ), padding = 'same' )
  lambdaLayers <- list( convolutionLayer )
  for( i in 1:2 )
    {
    splitLayer <- splitTensor3D( axis = 4, ratioSplit = 2, idSplit = i )
    lambdaLayers <- lappend( lambdaLayers, outputs %>% splitLayer )
    }
  outputs <- layer_concatenate( lambdaLayers )

  # Conv5
  outputs <- outputs %>% layer_zero_padding_3d( padding = c( 2, 2, 2 ) )
  normalizationLayer <- crossChannelNormalization3D()
  outputs <- outputs %>% normalizationLayer

  convolutionLayer <- outputs %>% layer_conv_3d( filters = 128, 
    kernel_size = c( 3, 3, 3 ), padding = 'same' )
  lambdaLayers <- list( convolutionLayer )
  for( i in 1:2 )
    {
    splitLayer <- splitTensor3D( axis = 4, ratioSplit = 2, idSplit = i )
    lambdaLayers <- lappend( lambdaLayers, outputs %>% splitLayer )
    }
  outputs <- layer_concatenate( lambdaLayers )

  outputs <- outputs %>% layer_max_pooling_3d( pool_size = c( 3, 3, 3 ), 
    strides = c( 2, 2, 2 ) )
  outputs <- outputs %>% layer_flatten()
  outputs <- outputs %>% layer_dense( units = denseUnits, activation = 'relu' )
  if( dropoutRate > 0.0 )
    {
    outputs <- outputs %>% layer_dropout( rate = dropoutRate )
    }
  outputs <- outputs %>% layer_dense( units = denseUnits, activation = 'relu' )
  if( dropoutRate > 0.0 )
    {
    outputs <- outputs %>% layer_dropout( rate = dropoutRate )
    }
  outputs <- outputs %>% layer_dense( units = numberOfClassificationLabels, activation = 'softmax' )

  alexNetModel <- keras_model( inputs = inputs, outputs = outputs )

  return( alexNetModel )
}

