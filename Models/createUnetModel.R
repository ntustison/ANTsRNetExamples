#' Model loss function for multilabel problems--- multilabel dice coefficient
#'  
#' Taken the keras loss function (losses.R):
#' 
#'    https://github.com/rstudio/keras/blob/master/R/losses.R
#' 
#' @param y_true True labels (Tensor) 
#' @param y_pred Predictions (Tensor of the same shape as `y_true`)
#' 
#' @details Loss functions are to be supplied in the `loss` parameter of the 
#' [compile()] function.
#' 
#' Loss functions can be specified either using the name of a built in loss
#' function (e.g. 'loss = binary_crossentropy'), a reference to a built in loss
#' function (e.g. 'loss = loss_binary_crossentropy()') or by passing an
#' artitrary function that returns a scalar for each data-point and takes the
#' following two arguments: 
#' 
#' - `y_true` True labels (Tensor) 
#' - `y_pred` Predictions (Tensor of the same shape as `y_true`)
#' 
#' The actual optimized objective is the mean of the output array across all
#' datapoints.

multilabel_dice_coefficient <- function( y_true, y_pred )
{
  smoothingFactor <- 1.0

  K <- keras::backend()  

  y_dims <- unlist( K$int_shape( y_pred ) )
  numberOfLabels <- y_dims[length( y_dims )]

  # Unlike native R, indexing starts at '0'.  However, we are 
  # assuming the background is 0 so we skip index 0.

  if( length( y_dims ) == 3 )
    {
    # 2-D image
    y_true_permuted <- K$permute_dimensions( y_true, pattern = c( 3L, 0L, 1L, 2L ) )
    y_pred_permuted <- K$permute_dimensions( y_pred, pattern = c( 3L, 0L, 1L, 2L ) )
    } else {
    # 3-D image  
    y_true_permuted <- K$permute_dimensions( y_true, pattern = c( 4L, 0L, 1L, 2L, 3L ) )
    y_pred_permuted <- K$permute_dimensions( y_pred, pattern = c( 4L, 0L, 1L, 2L, 3L ) )
    }
  y_true_label <- K$gather( y_true_permuted, indices = c( 1L ) )
  y_pred_label <- K$gather( y_pred_permuted, indices = c( 1L ) )

  y_true_label_f <- K$flatten( y_true_label )
  y_pred_label_f <- K$flatten( y_pred_label )
  intersection <-  y_true_label_f * y_pred_label_f
  union <- y_true_label_f + y_pred_label_f - intersection

  numerator <- K$sum( intersection )
  denominator <- K$sum( union )

  if( numberOfLabels > 2 )
    {
    for( j in 2L:( numberOfLabels - 1L ) )
      {
      y_true_label <- K$gather( y_true_permuted, indices = c( j ) )
      y_pred_label <- K$gather( y_pred_permuted, indices = c( j ) )
      y_true_label_f <- K$flatten( y_true_label )
      y_pred_label_f <- K$flatten( y_pred_label )
      intersection <-  y_true_label_f * y_pred_label_f
      union <- y_true_label_f + y_pred_label_f - intersection

      numerator <- numerator + K$sum( intersection )
      denominator <- denominator + K$sum( union )
      }
    }  
  unionOverlap <- numerator / denominator 

  return ( ( 2.0 * unionOverlap + smoothingFactor ) / ( 1.0 + unionOverlap + smoothingFactor ) )
}
attr( multilabel_dice_coefficient, "py_function_name" ) <- "multilabel_dice_coefficient"

loss_multilabel_dice_coefficient_error <- function( y_true, y_pred )
{
  return( -multilabel_dice_coefficient( y_true, y_pred ) )
}
attr( loss_multilabel_dice_coefficient_error, "py_function_name" ) <- "multilabel_dice_coefficient_error"

#' 2-D implementation of the U-net deep learning architecture.
#'
#' Creates a keras model of the U-net deep learning architecture for image 
#' segmentation.  More information is provided at the authors' website:
#' 
#'         https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
#' 
#' with the paper available here:
#' 
#'         https://arxiv.org/abs/1505.04597
#'
#' This particular implementation was influenced by the following python 
#' implementation: 
#' 
#'         https://github.com/joelthelion/ultrasound-nerve-segmentation       
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).  The batch size
#' (i.e., number of training images) is not specified a priori. 
#' @param numberOfClassificationLabels Number of segmentation labels.  
#' @param layers a vector determining the number of 'filters' defined at
#' for each layer.
#' @param lowestResolution number of filters at the beginning and end of 
#' the 'U'.
#' @param convolutionKernelSize 2-d vector definining the kernel size 
#' during the encoding path
#' @param deconvolutionKernelSize 2-d vector definining the kernel size 
#' during the decoding 
#' @param poolSize 2-d vector defining the region for each pooling layer.
#' @param strides 2-d vector describing the stride length in each direction.
#'
#' @return a u-net keras model to be used with subsequent fitting
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
#'  unetModel <- createUnetModel2D( dim( trainingImageArrays[[1]] ), 
#'    numberOfClassificationLabels = numberOfLabels, layers = 1:4 )
#'  
#'  # Fit the model
#'  
#'  track <- unetModel %>% fit( X_train, Y_train, 
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

createUnetModel2D <- function( inputImageSize, 
                               numberOfClassificationLabels = 1,
                               layers = 1:4, 
                               lowestResolution = 32, 
                               convolutionKernelSize = c( 3, 3 ), 
                               deconvolutionKernelSize = c( 2, 2 ), 
                               poolSize = c( 2, 2 ), 
                               strides = c( 2, 2 )
                             )
{
  if ( ! usePkg( "keras" ) )
    {
    stop( "Please install the keras package." )
    }

  inputs <- layer_input( shape = inputImageSize )

  # Encoding path  

  encodingConvolutionLayers <- list()
  for( i in 1:length( layers ) )
    {
    numberOfFilters <- lowestResolution * 2 ^ ( layers[i] - 1 )

    if( i == 1 )
      {
      conv <- inputs %>% layer_conv_2d( filters = numberOfFilters, 
        kernel_size = convolutionKernelSize, activation = 'relu', padding = 'same' )
      } else {
      conv <- pool %>% layer_conv_2d( filters = numberOfFilters, 
        kernel_size = convolutionKernelSize, activation = 'relu', padding = 'same' )
      }
    encodingConvolutionLayers[[i]] <- conv %>% layer_conv_2d( 
      filters = numberOfFilters, kernel_size = convolutionKernelSize, 
      activation = 'relu', padding = 'same' )
    
    if( i < length( layers ) )
      {
      pool <- encodingConvolutionLayers[[i]] %>% 
        layer_max_pooling_2d( pool_size = poolSize, strides = strides )
      }
    }

  # Decoding path 

  outputs <- encodingConvolutionLayers[[length( layers )]]
  for( i in 2:length( layers ) )
    {
    numberOfFilters <- lowestResolution * 2 ^ ( length( layers ) - layers[i] )    
    outputs <- layer_concatenate( list( outputs %>%  
      layer_conv_2d_transpose( filters = numberOfFilters, 
        kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same' ),
      encodingConvolutionLayers[[length( layers ) - i + 1]] ),
      axis = 3
      )

    outputs <- outputs %>% layer_conv_2d( filters = numberOfFilters, 
      kernel_size = convolutionKernelSize, activation = 'relu', padding = 'same' )  %>%
      layer_conv_2d( filters = numberOfFilters, kernel_size = convolutionKernelSize, 
      activation = 'relu', padding = 'same'  )  
    }
  if( numberOfClassificationLabels == 1 )  
    {
    outputs <- outputs %>% layer_conv_2d( filters = numberOfClassificationLabels, 
      kernel_size = c( 1, 1 ), activation = 'sigmoid' )
    } else {
    outputs <- outputs %>% layer_conv_2d( filters = numberOfClassificationLabels, 
      kernel_size = c( 1, 1 ), activation = 'softmax' )
    }
    
  unetModel <- keras_model( inputs = inputs, outputs = outputs )

  return( unetModel )
}

  
#' 3-D image segmentation implementation of the U-net deep learning architecture.
#'
#' Creates a keras model of the U-net deep learning architecture for image 
#' segmentation.  More information is provided at the authors' website:
#' 
#'         https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
#' 
#' with the paper available here:
#' 
#'         https://arxiv.org/abs/1505.04597
#'
#' This particular implementation was influenced by the following python 
#' implementation: 
#' 
#'         https://github.com/joelthelion/ultrasound-nerve-segmentation       
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).  The batch size
#' (i.e., number of training images) is not specified a priori. 
#' @param numberOfClassificationLabels Number of segmentation labels.  
#' @param layers a vector determining the number of 'filters' defined at
#' for each layer.
#' @param lowestResolution number of filters at the beginning and end of 
#' the 'U'.
#' @param convolutionKernelSize 3-d vector definining the kernel size 
#' during the encoding path
#' @param deconvolutionKernelSize 3-d vector definining the kernel size 
#' during the decoding 
#' @param poolSize 3-d vector defining the region for each pooling layer.
#' @param strides 3-d vector describing the stride length in each direction.
#'
#' @return a u-net keras model to be used with subsequent fitting
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
#'  # Create the model (3-D function is a straightforward analog)
#'  
#'  unetModel <- createUnetModel2D( dim( trainingImageArrays[[1]] ), 
#'    numberOfClassificationLabels = numberOfLabels, layers = 1:4 )
#'  
#'  # Fit the model
#'  
#'  track <- unetModel %>% fit( X_train, Y_train, 
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

createUnetModel3D <- function( inputImageSize, 
                               numberOfClassificationLabels = 1,
                               layers = 1:4, 
                               lowestResolution = 32, 
                               convolutionKernelSize = c( 3, 3, 3 ), 
                               deconvolutionKernelSize = c( 2, 2, 2 ), 
                               poolSize = c( 2, 2, 2 ), 
                               strides = c( 2, 2, 2 )
                             )
{

  if ( ! usePkg( "keras" ) )
    {
    stop( "Please install the keras package." )
    }

  inputs <- layer_input( shape = c( inputImageSize, numberOfClassificationLabels ) )

  # Encoding path  

  encodingConvolutionLayers <- list()
  for( i in 1:length( layers ) )
    {
    numberOfFilters <- lowestResolution * 2 ^ ( layers[i] - 1 )

    if( i == 1 )
      {
      conv <- inputs %>% layer_conv_3d( filters = numberOfFilters, 
        kernel_size = convolutionKernelSize, activation = 'relu', padding = 'same' )
      } else {
      conv <- pool %>% layer_conv_3d( filters = numberOfFilters, 
        kernel_size = convolutionKernelSize, activation = 'relu', padding = 'same' )
      }
    encodingConvolutionLayers[[i]] <- conv %>% layer_conv_3d( 
      filters = numberOfFilters, kernel_size = convolutionKernelSize, 
      activation = 'relu', padding = 'same' )
    
    if( i < length( layers ) )
      {
      pool <- encodingConvolutionLayers[[i]] %>% 
        layer_max_pooling_3d( pool_size = poolSize, strides = strides )
      }
    }

  # Decoding path 

  outputs <- encodingConvolutionLayers[[length( layers )]]
  for( i in 2:length( layers ) )
    {
    numberOfFilters <- lowestResolution * 2 ^ ( length( layers ) - layers[i] )    
    outputs <- layer_concatenate( list( outputs %>%  
      layer_conv_3d_transpose( filters = numberOfFilters, 
        kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same' ),
      encodingConvolutionLayers[[length( layers ) - i + 1]] ),
      axis = 3
      )

    outputs <- outputs %>% layer_conv_3d( filters = numberOfFilters, 
      kernel_size = convolutionKernelSize, activation = 'relu', padding = 'same'  )  %>%
      layer_conv_3d( filters = numberOfFilters, 
        kernel_size = convolutionKernelSize, activation = 'relu', padding = 'same'  )  
    }
  if( numberOfClassificationLabels == 1 )  
    {
    outputs <- outputs %>% layer_conv_2d( filters = numberOfClassificationLabels, 
      kernel_size = c( 1, 1, 1 ), activation = 'sigmoid' )
    } else {
    outputs <- outputs %>% layer_conv_2d( filters = numberOfClassificationLabels, 
      kernel_size = c( 1, 1, 1 ), activation = 'softmax' )
    }

  unetModel <- keras_model( inputs = inputs, outputs = outputs )

  return( unetModel )
}
  