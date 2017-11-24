#' 2-D implementation of the GoogLeNet deep learning architecture.
#'
#' Creates a keras model of the GoogLeNet deep learning architecture for image 
#' recognition based on the paper
#' 
#' C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, 
#'   A. Rabinovich, Going Deeper with Convolutions
#' C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna. Rethinking the Inception 
#'   Architecture for Computer Vision
#' 
#' available here:
#' 
#'         https://arxiv.org/abs/1409.4842
#'         https://arxiv.org/abs/1512.00567
#'
#' This particular implementation was influenced by the following python 
#' implementation: 
#' 
#'         https://github.com/fchollet/deep-learning-models/blob/master/inception_v3.py     
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).  The batch size
#' (i.e., number of training images) is not specified a priori. 
#' @param numberOfClassificationLabels Number of segmentation labels.  
#'
#' @return a GoogLeNet keras model to be used with subsequent fitting
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
#'  outputs <- createGoogLeNetModel2D( dim( trainingImageArrays[[1]] ), 
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

createGoogLeNetModel2D <- function( inputImageSize, 
                                    numberOfClassificationLabels = 1000,
                                    dropoutRate = 0.0
                                  )
{

  if ( ! usePkg( "keras" ) )
    {
    stop( "Please install the keras package." )
    }

  
  convolutionAndBatchNormalization2D <- function( model, 
                                                  numberOfFilters, 
                                                  kernelSize, 
                                                  padding = 'same',
                                                  strides = c( 1, 1 ) )
    {
    K <- keras::backend()

    channelAxis <- 1  
    if( K$image_data_format() == 'channels_last' )
      {
      channelAxis <- 3 
      }

    model <- model %>% layer_conv_2d( numberOfFilters, 
      kernel_size = kernelSize, padding = padding, strides = strides, 
      use_bias = TRUE )
    model <- model %>% layer_batch_normalization( axis = channelAxis, 
      scale = FALSE )  
    model <- model %>% layer_activation( activation = 'relu' )

    return( model )
    }                                                    

  K <- keras::backend()
  channelAxis <- 1  
  if( K$image_data_format() == 'channels_last' )
    {
    channelAxis <- 3 
    }

  inputs <- layer_input( shape = inputImageSize )
  
  outputs <- convolutionAndBatchNormalization2D( inputs, numberOfFilters = 32, 
    kernelSize = c( 3, 3 ), strides = c( 2, 2 ), padding = 'valid' )
  outputs <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 32,
    kernelSize = c( 3, 3 ), padding = 'valid' )
  outputs <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 64,
    kernelSize = c( 3, 3 ) )
  outputs <- outputs %>% layer_max_pooling_2d( pool_size = c( 3, 3 ), 
    strides = c( 2, 2 ) )  

  outputs <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 80,
    kernelSize = c( 1, 1 ), padding = 'valid' )
  outputs <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 192,
    kernelSize = c( 3, 3 ) )
  outputs <- outputs %>% layer_max_pooling_2d( pool_size = c( 3, 3 ), 
    strides = c( 2, 2 ) )  

  # mixed 0, 1, 2: 35x35x256
  branchLayers <- list()
  branchLayers[[1]] <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 64,
    kernelSize = c( 1, 1 ) )
  branchLayers[[2]] <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 48,
    kernelSize = c( 1, 1 ) )
  branchLayers[[2]] <- convolutionAndBatchNormalization2D( branchLayers[[2]], 
    numberOfFilters = 64, kernelSize = c( 5, 5 ) )
  branchLayers[[3]] <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 64, 
    kernelSize = c( 1, 1 ) )
  branchLayers[[3]] <- convolutionAndBatchNormalization2D( branchLayers[[3]], 
    numberOfFilters = 96, kernelSize = c( 3, 3 ) )
  branchLayers[[3]] <- convolutionAndBatchNormalization2D( branchLayers[[3]], 
    numberOfFilters = 96, kernelSize = c( 3, 3 ) )
  branchLayers[[4]] <- outputs %>% layer_average_pooling_2d( pool_size = c( 3, 3 ), 
    strides = c( 1, 1 ), padding = 'same' )
  branchLayers[[4]] <- convolutionAndBatchNormalization2D( branchLayers[[4]], 
    numberOfFilters = 32, kernelSize = c( 1, 1 ) )  
  outputs <- layer_concatenate( branchLayers, axis = channelAxis )

  # mixed 1: 35x35x256
  branchLayers <- list()
  branchLayers[[1]] <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 64,
    kernelSize = c( 1, 1 ) )
  branchLayers[[2]] <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 48,
    kernelSize = c( 1, 1 ) )
  branchLayers[[2]] <- convolutionAndBatchNormalization2D( branchLayers[[2]], 
    numberOfFilters = 64, kernelSize = c( 5, 5 ) )
  branchLayers[[3]] <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 64, 
    kernelSize = c( 1, 1 ) )
  branchLayers[[3]] <- convolutionAndBatchNormalization2D( branchLayers[[3]], 
    numberOfFilters = 96, kernelSize = c( 3, 3 ) )
  branchLayers[[3]] <- convolutionAndBatchNormalization2D( branchLayers[[3]], 
    numberOfFilters = 96, kernelSize = c( 3, 3 ) )
  branchLayers[[4]] <- outputs %>% layer_average_pooling_2d( pool_size = c( 3, 3 ), 
    strides = c( 1, 1 ), padding = 'same' )
  branchLayers[[4]] <- convolutionAndBatchNormalization2D( branchLayers[[4]], 
    numberOfFilters = 32, kernelSize = c( 1, 1 ) )  
  outputs <- layer_concatenate( branchLayers, axis = channelAxis )

  # mixed 2: 35x35x256
  branchLayers <- list()
  branchLayers[[1]] <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 64,
    kernelSize = c( 1, 1 ) )
  branchLayers[[2]] <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 48,
    kernelSize = c( 1, 1 ) )
  branchLayers[[2]] <- convolutionAndBatchNormalization2D( branchLayers[[2]], 
    numberOfFilters = 64, kernelSize = c( 5, 5 ) )
  branchLayers[[3]] <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 64, 
    kernelSize = c( 1, 1 ) )
  branchLayers[[3]] <- convolutionAndBatchNormalization2D( branchLayers[[3]], 
    numberOfFilters = 96, kernelSize = c( 3, 3 ) )
  branchLayers[[3]] <- convolutionAndBatchNormalization2D( branchLayers[[3]], 
    numberOfFilters = 96, kernelSize = c( 3, 3 ) )
  branchLayers[[4]] <- outputs %>% layer_average_pooling_2d( pool_size = c( 3, 3 ), 
    strides = c( 1, 1 ), padding = 'same' )
  branchLayers[[4]] <- convolutionAndBatchNormalization2D( branchLayers[[4]], 
    numberOfFilters = 32, kernelSize = c( 1, 1 ) )  
  outputs <- layer_concatenate( branchLayers, axis = channelAxis )
  
  # mixed 3: 17x17x768
  branchLayers <- list()
  branchLayers[[1]] <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 384,
    kernelSize = c( 3, 3 ), strides = c( 2, 2 ), padding = 'valid' )
  branchLayers[[2]] <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 64,
    kernelSize = c( 1, 1 ) )
  branchLayers[[2]] <- convolutionAndBatchNormalization2D( branchLayers[[2]], 
    numberOfFilters = 96, kernelSize = c( 3, 3 ) )
  branchLayers[[2]] <- convolutionAndBatchNormalization2D( branchLayers[[2]], 
    numberOfFilters = 96, kernelSize = c( 3, 3 ), strides = c( 2, 2 ), padding = 'valid' )
  branchLayers[[3]] <- outputs %>% layer_max_pooling_2d( pool_size = c( 3, 3 ), 
    strides = c( 2, 2 ) )
  outputs <- layer_concatenate( branchLayers, axis = channelAxis )

  # mixed 4: 17x17x768
  branchLayers <- list()
  branchLayers[[1]] <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 192,
    kernelSize = c( 1, 1 ) )
  branchLayers[[2]] <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 128,
    kernelSize = c( 1, 1 ) )
  branchLayers[[2]] <- convolutionAndBatchNormalization2D( branchLayers[[2]], 
    numberOfFilters = 128, kernelSize = c( 1, 7 ) )
  branchLayers[[2]] <- convolutionAndBatchNormalization2D( branchLayers[[2]], 
    numberOfFilters = 192, kernelSize = c( 7, 1 ) )
  branchLayers[[3]] <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 128,
    kernelSize = c( 1, 1 ) )
  branchLayers[[3]] <- convolutionAndBatchNormalization2D( branchLayers[[3]], 
    numberOfFilters = 128, kernelSize = c( 7, 1 ) )
  branchLayers[[3]] <- convolutionAndBatchNormalization2D( branchLayers[[3]], 
    numberOfFilters = 128, kernelSize = c( 1, 7 ) )
  branchLayers[[3]] <- convolutionAndBatchNormalization2D( branchLayers[[3]], 
    numberOfFilters = 128, kernelSize = c( 7, 1 ) )
  branchLayers[[3]] <- convolutionAndBatchNormalization2D( branchLayers[[3]], 
    numberOfFilters = 192, kernelSize = c( 1, 7 ) )
  branchLayers[[4]] <- outputs %>% layer_average_pooling_2d( pool_size = c( 3, 3 ), 
    strides = c( 1, 1 ), padding = 'same' )
  branchLayers[[4]] <- convolutionAndBatchNormalization2D( branchLayers[[4]], 
    numberOfFilters = 192, kernelSize = c( 1, 1 ) )  
  outputs <- layer_concatenate( branchLayers, axis = channelAxis )
    
  # mixed 4: 17x17x768
  for( i in 1:2 )  
    {
    branchLayers <- list()
    branchLayers[[1]] <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 192,
      kernelSize = c( 1, 1 ) )
    branchLayers[[2]] <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 160,
      kernelSize = c( 1, 1 ) )
    branchLayers[[2]] <- convolutionAndBatchNormalization2D( branchLayers[[2]], 
      numberOfFilters = 160, kernelSize = c( 1, 7 ) )
    branchLayers[[2]] <- convolutionAndBatchNormalization2D( branchLayers[[2]], 
      numberOfFilters = 192, kernelSize = c( 7, 1 ) )
    branchLayers[[3]] <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 160,
      kernelSize = c( 1, 1 ) )
    branchLayers[[3]] <- convolutionAndBatchNormalization2D( branchLayers[[3]], 
      numberOfFilters = 160, kernelSize = c( 7, 1 ) )
    branchLayers[[3]] <- convolutionAndBatchNormalization2D( branchLayers[[3]], 
      numberOfFilters = 160, kernelSize = c( 1, 7 ) )
    branchLayers[[3]] <- convolutionAndBatchNormalization2D( branchLayers[[3]], 
      numberOfFilters = 160, kernelSize = c( 7, 1 ) )
    branchLayers[[3]] <- convolutionAndBatchNormalization2D( branchLayers[[3]], 
      numberOfFilters = 192, kernelSize = c( 1, 7 ) )
    branchLayers[[4]] <- outputs %>% layer_average_pooling_2d( pool_size = c( 3, 3 ), 
      strides = c( 1, 1 ), padding = 'same' )
    branchLayers[[4]] <- convolutionAndBatchNormalization2D( branchLayers[[4]], 
      numberOfFilters = 192, kernelSize = c( 1, 1 ) )  
    outputs <- layer_concatenate( branchLayers, axis = channelAxis )
    }

  # mixed 7: 17x17x768
  branchLayers <- list()
  branchLayers[[1]] <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 192,
    kernelSize = c( 1, 1 ) )
  branchLayers[[2]] <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 192,
    kernelSize = c( 1, 1 ) )
  branchLayers[[2]] <- convolutionAndBatchNormalization2D( branchLayers[[2]], 
    numberOfFilters = 192, kernelSize = c( 1, 7 ) )
  branchLayers[[2]] <- convolutionAndBatchNormalization2D( branchLayers[[2]], 
    numberOfFilters = 192, kernelSize = c( 7, 1 ) )
  branchLayers[[3]] <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 192,
    kernelSize = c( 1, 1 ) )
  branchLayers[[3]] <- convolutionAndBatchNormalization2D( branchLayers[[3]], 
    numberOfFilters = 192, kernelSize = c( 7, 1 ) )
  branchLayers[[3]] <- convolutionAndBatchNormalization2D( branchLayers[[3]], 
    numberOfFilters = 192, kernelSize = c( 1, 7 ) )
  branchLayers[[3]] <- convolutionAndBatchNormalization2D( branchLayers[[3]], 
    numberOfFilters = 192, kernelSize = c( 7, 1 ) )
  branchLayers[[3]] <- convolutionAndBatchNormalization2D( branchLayers[[3]], 
    numberOfFilters = 192, kernelSize = c( 1, 7 ) )
  branchLayers[[4]] <- outputs %>% layer_average_pooling_2d( pool_size = c( 3, 3 ), 
    strides = c( 1, 1 ), padding = 'same' )
  branchLayers[[4]] <- convolutionAndBatchNormalization2D( branchLayers[[4]], 
    numberOfFilters = 192, kernelSize = c( 1, 1 ) )  
  outputs <- layer_concatenate( branchLayers, axis = channelAxis )

  # mixed 8: 8x8x1280
  branchLayers <- list()
  branchLayers[[1]] <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 192,
    kernelSize = c( 1, 1 ) )
  branchLayers[[1]] <- convolutionAndBatchNormalization2D( branchLayers[[1]], 
    numberOfFilters = 320, kernelSize = c( 3, 3 ), strides = c( 2, 2 ), 
    padding = 'valid' )
  branchLayers[[2]] <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 192,
    kernelSize = c( 1, 1 ) )    
  branchLayers[[2]] <- convolutionAndBatchNormalization2D( branchLayers[[2]], 
    numberOfFilters = 192, kernelSize = c( 1, 7 ) )
  branchLayers[[2]] <- convolutionAndBatchNormalization2D( branchLayers[[2]], 
    numberOfFilters = 192, kernelSize = c( 7, 1 ) )
  branchLayers[[2]] <- convolutionAndBatchNormalization2D( branchLayers[[2]], 
    numberOfFilters = 192, kernelSize = c( 3, 3 ), strides = c( 2, 2 ), 
    padding = 'valid' )
  branchLayers[[3]] <- outputs %>% layer_max_pooling_2d( pool_size = c( 3, 3 ), 
    strides = c( 2, 2 ) )
  outputs <- layer_concatenate( branchLayers, axis = channelAxis )

  # mixed 9: 8x8x2048
  for( i in 1:2 )  
    {
    branchLayers <- list()

    branchLayer <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 320,
      kernelSize = c( 1, 1 ) )
    branchLayers[[1]] <- branchLayer  

    branchLayer <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 384,
      kernelSize = c( 1, 1 ) )
    branchLayer1 <- convolutionAndBatchNormalization2D( branchLayer, 
      numberOfFilters = 384, kernelSize = c( 1, 3 ) )
    branchLayer2 <- convolutionAndBatchNormalization2D( branchLayer, 
      numberOfFilters = 384, kernelSize = c( 3, 1 ) )
    branchLayers[[2]] <- layer_concatenate( list( branchLayer1, branchLayer2 ), 
      axis = channelAxis )

    branchLayer <- convolutionAndBatchNormalization2D( outputs, numberOfFilters = 448,
      kernelSize = c( 1, 1 ) )
    branchLayer <- convolutionAndBatchNormalization2D( branchLayer, 
      numberOfFilters = 384, kernelSize = c( 3, 3 ) )
    branchLayer1 <- convolutionAndBatchNormalization2D( branchLayer, 
      numberOfFilters = 384, kernelSize = c( 1, 3 ) )
    branchLayer2 <- convolutionAndBatchNormalization2D( branchLayer, 
      numberOfFilters = 384, kernelSize = c( 3, 1 ) )
    branchLayers[[3]] <- layer_concatenate( list( branchLayer1, branchLayer2 ), 
      axis = channelAxis )
    
    branchLayers[[4]] <- outputs %>% layer_average_pooling_2d( pool_size = c( 3, 3 ), 
      strides = c( 1, 1 ), padding = 'same' )
    branchLayers[[4]] <- convolutionAndBatchNormalization2D( branchLayers[[4]], 
      numberOfFilters = 192, kernelSize = c( 1, 1 ) )  

    outputs <- layer_concatenate( branchLayers, axis = channelAxis )  
    }
  outputs <- outputs %>% layer_global_average_pooling_2d()
  outputs <- outputs %>% layer_dense( units = numberOfClassificationLabels, 
    activation = 'softmax' )

  googLeNetModel <- keras_model( inputs = inputs, outputs = outputs )

  return( googLeNetModel )
}





