#' 2-D implementation of the AlexNet deep learning architecture.
#'
#' Creates a keras model of the AlexNet deep learning architecture for image 
#' recognition based on the paper
#' 
#' K. Simonyan and A. Zisserman, ImageNet Classification with Deep Convolutional Neural 
#'   Networks
#' 
#' available here:
#' 
#'         http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
#'
#' This particular implementation was influenced by the following python 
#' implementation: 
#' 
#'         https://github.com/duggalrahul/AlexNet-Experiments-Keras/blob/master/Code/alexnet_base.py      
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
#'  alexNetModel <- createAlexNetModel2D( dim( trainingImageArrays[[1]] ), 
#'    numberOfClassificationLabels = numberOfLabels, layers = 1:4 )
#'  
#'  # Fit the model
#'  
#'  track <- alexNetModel %>% fit( X_train, Y_train, 
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
                                  numberOfClassificationLabels = 1000
                                )
{

  if ( ! usePkg( "keras" ) )
    {
    stop( "Please install the keras package." )
    }

  splitTensor2D <- function( model, axis = 1, ratioSplit = 1, idSplit = 0  )
    {
    div <- as.integer( ( model$shape ) / ratioSplit )

    if( axis == 0 ) 
      {
      output <- model[( idSplit * div ):( ( idSplit + 1 ) * div ),,,]
      } else if( axis == 1 ) {
      output <- model[, ( idSplit * div ):( ( idSplit + 1 ) * div ),,]
      } else if( axis == 2 ) {
      output <- model[,, ( idSplit * div ):( ( idSplit + 1 ) * div ),]
      } else if( axis == 3 ) {
      output <- model[,,, ( idSplit * div ):( ( idSplit + 1 ) * div )]
      } else {
      stop( "Wrong axis specification.")  
      }
    return( output )
    }

  crossChannelNormalization <- function( alpha = 1e-4, k = 2, beta = 0.75, n = 5 )
    {

    normalizeTensor <- function( X )
      {
      K <- keras::backend() 
      half <- as.integer( n / 2 )
      X2 <- K$square( X )
      extraChannels <- K$spatial_2d_padding( 
        K$permute_dimensions( X2, pattern = c( 0L, 2L, 3L, 1L ) ), c( 0, half ) )
      extraChannels <- K$permute_dimensions( extraChannels, pattern = c( 0L, 3L, 1L, 2L ) )
      scale <- k

      ch <- unlist( K$int_shape( X ) )[2]
      for( i in 0:n )
        {
        scale <- scale + alpha * extraChannels[, i:( i + ch ),,]  
        }
      scale <- scale^beta

      X <- X / scale

      return( X )  
      }

    return( layer_lambda( normalizeTensor ) )  
    }

  inputs <- layer_input( shape = inputImageSize )

  # Conv1
  alexNetModel <- inputs %>% layer_conv_2d( numberOfFilters = 96, 
    kernel_size = c( 11, 11 ), strides = c( 4, 4 ), activation = 'relu', 
    kernel_initializer = "initializer_he_normal" )

  # Conv2
  alexNetModel <- alexNetModel %>% layer_max_pooling_2d( poolSize = c( 3, 3 ), 
    strides = c( 2, 2 ) )
  # Cross channel normalization  
  alexNetModel <- alexNetModel %>% createCrossChannelNormalization()

  alexNetModel <- alexNetModel %>% layer_zero_padding_2D( padding = c( 2, 2 ) )

  convolutionLayer <- alexNetModel %>% layer_conv_2d( numberOfFilters = 128 )
  lambdaLayers <- list( convolutionLayer )
  for( i in 0:2 )
    {
    lambdaLayers <- lappend( lambdaLayers, 
      splitTensor2D( alexNetModel, ratioSplit = 2, idSplit = i ) )
    }
  alexNetModel <- layer_concatenate( lambdaLayers )

  # Conv3
  alexNetModel <- alexNetModel %>% layer_max_pooling_2d( poolSize = c( 3, 3 ), 
    strides = c( 2, 2 ) )
  # Cross channel normalization  
  alexNetModel <- alexNetModel %>% createCrossChannelNormalization()

  alexNetModel <- alexNetModel %>% layer_zero_padding_2D( padding = c( 2, 2 ) )
  alexNetModel <- alexNetModel %>% layer_conv_2d( numberOfFilters = 384 )

  # Conv4
  alexNetModel <- alexNetModel %>% layer_zero_padding_2D( padding = c( 2, 2 ) )
  convolutionLayer <- alexNetModel %>% layer_conv_2d( numberOfFilters = 128 )
  lambdaLayers <- list( convolutionLayer )
  for( i in 0:2 )
    {
    lambdaLayers <- lappend( lambdaLayers, 
      splitTensor2D( alexNetModel, ratioSplit = 2, idSplit = i ) )
    }
  alexNetModel <- layer_concatenate( lambdaLayers )

  # Conv5
  alexNetModel <- alexNetModel %>% layer_zero_padding_2D( padding = c( 2, 2 ) )
  # Cross channel normalization  
  alexNetModel <- alexNetModel %>% createCrossChannelNormalization()

  convolutionLayer <- alexNetModel %>% layer_conv_2d( numberOfFilters = 128 )
  lambdaLayers <- list( convolutionLayer )
  for( i in 0:2 )
    {
    lambdaLayers <- lappend( lambdaLayers, 
      splitTensor2D( alexNetModel, ratioSplit = 2, idSplit = i ) )
    }
  alexNetModel <- layer_concatenate( lambdaLayers )


  alexNetModel %>% layer_max_pooling_2d( pool_size = c( 3, 3 ), strides = c(2, 2 ) )
  alexNetModel %>% layer_flatten()
  alexNetModel %>% layer_dense( units = denseUnits, activation = 'relu' )
  if( dropoutRate > 0.0 )
    {
    alexNetModel %>% layer_dropout( rate = dropoutRate )
    }
  alexNetModel %>% layer_dense( units = denseUnits, activation = 'relu' )
  if( dropoutRate > 0.0 )
    {
    alexNetModel %>% layer_dropout( rate = dropoutRate )
    }
  alexNetModel %>% layer_dense( units = numberOfClassificationLabels, activation = 'softmax' )

  return( alexNetModel )
}

