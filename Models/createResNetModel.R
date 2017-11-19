#' 2-D implementation of the ResNet deep learning architecture.
#'
#' Creates a keras model of the ResNet deep learning architecture for image 
#' classification.  The paper is available here:
#' 
#'         https://arxiv.org/abs/1512.03385
#'
#' This particular implementation was influenced by the following python 
#' implementation: 
#' 
#'         https://gist.github.com/mjdietzx/0cb95922aac14d446a6530f87b3a04ce    
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
#' @return a ResNet keras model to be used with subsequent fitting
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

createResNetModel2D <- function( inputImageSize, 
                                 numberOfClassificationLabels = 1000,
                                 layers = 1:4, 
                                 residualBlockSchedule = c( 4, 5, 7, 4 ),
                                 lowestResolution = 64,
                                 cardinality = 32
                               )
{
  if ( ! usePkg( "keras" ) )
    {
    stop( "Please install the keras package." )
    }

  addCommonLayers <- function( model )
    {
    model <- model %>% layer_batch_normalization()
    model <- model %>% layer_activation_leaky_relu()

    return( model )
    }

  groupedConvolutionLayer2D <- function( model, numberOfFilters, strides )
    {

    # Per standard ResNet, this is just a 2-D convolution
    if( cardinality == 1 )
      {
      model %>% layer_conv_2d( filters = numberOfFilters, 
        kernel_size = c( 3, 3 ), padding = 'same' )
      return( model )
      }

    if( numberOfFilters %% cardinality != 0 )  
      {
      stop( "numberOfFilters %% cardinality != 0" )  
      }

    numberOfGroupFilters <- as.integer( numberOfFilters / cardinality )

    convolutionLayers <- list()
    for( j in 1:cardinality )
      {
      convolutionLayers[[j]] <- model %>% layer_lambda( function( z ) 
        { z[,,, ( j * numberOfGroupFilters ):( ( j + 1 ) * numberOfGroupFilters )] } )
      convolutionLayers[[j]] <- convolutionLayers[[j]] %>% 
        layer_conv_2d( filters = numberOfGroupFilters, 
          kernel_size = c( 3, 3 ), strides = strides, padding = 'same' )
      }

    return( layer_concatenate( convolutionLayers ) )
    }

  residualBlock2D <- function( model, numberOfFiltersIn, numberOfFiltersOut, 
    strides = c( 1, 1 ), projectShortcut = FALSE )
    {
    shortcut <- model

    model <- model %>% layer_conv_2d( filters = numberOfFiltersIn, 
      kernel_size = c( 1, 1 ), strides = c( 1, 1 ), padding = 'same' )
    model <- addCommonLayers( model )

    # ResNeXt (identical to ResNet when `cardinality` == 1)
    model <- groupedConvolutionLayer2D( model, numberOfFilters = numberOfFiltersIn, 
      strides = strides )
    model <- addCommonLayers( model ) 

    model <- model %>% layer_conv_2d( filters = numberOfFiltersOut, 
      kernel_size = c( 1, 1 ), strides = c( 1, 1 ), padding = 'same' )
    model <- model %>% layer_batch_normalization() 

    if( projectShortcut == TRUE || prod( strides == c( 1, 1 ) ) == 0 )
      {
      shortcut <- shortcut %>% layer_conv_2d( filters = numberOfFiltersOut, 
        kernel_size = c( 1, 1 ), strides = strides, padding = 'same' )
      shortcut <- shortcut %>% layer_batch_normalization()  
      }

    model <- layer_add( list( shortcut, model ) )

    model <- model %>% layer_activation_leaky_relu()

    return( model )
    }

  inputs <- layer_input( shape = inputImageSize )

  # Convolution 1
  nFilters <- lowestResolution

  outputs <- inputs %>% layer_conv_2d( filters = nFilters, 
    kernel_size = c( 7, 7 ), strides = c( 2, 2 ) )
  outputs <- addCommonLayers( outputs )  
  outputs <- outputs %>% layer_max_pooling_2d( pool_size = c( 3, 3 ), 
    strides = c( 2, 2 ), padding = 'same' )

  for( i in 1:length( layers ) )
    {
    nFiltersIn <- lowestResolution * 2 ^ ( layers[i] )
    nFiltersOut <- 2 * nFiltersIn
    cat( i, "\n" )
    for( j in 1:residualBlockSchedule[i] )  
      {
      cat( "  ", j, " out of ", residualBlockSchedule[i], "\n" )
      projectShortcut <- FALSE
      if( i == 1 && j == 1 )  
        {
        projectShortcut <- TRUE  
        }
      if( i > 1 && j == 1 )
        {
        strides <- c( 2, 2 )
        } else {
        strides <- c( 1, 1 )  
        }
      outputs <- residualBlock2D( outputs, numberOfFiltersIn = nFiltersIn, 
        numberOfFiltersOut = nFiltersOut, strides = strides, 
        projectShortcut = projectShortcut )  
      }
    }  
  outputs <- outputs %>% layer_global_average_pooling_2d()
  outputs <- outputs %>% layer_dense( 1 )

  resNetModel <- keras_model( inputs = inputs, outputs = outputs )

  return( resNetModel )
}