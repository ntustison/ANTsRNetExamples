#' 2-D implementation of the DenseNet deep learning architecture.
#'
#' Creates a keras model of the DenseNet deep learning architecture for image 
#' recognition based on the paper
#' 
#' G. Huang, Z. Liu, K. Weinberger, and L. van der Maaten. Densely Connected 
#'   Convolutional Networks Networks
#' 
#' available here:
#' 
#'         https://arxiv.org/abs/1608.06993
#'
#' This particular implementation was influenced by the following python 
#' implementation: 
#' 
#'         https://github.com/tdeboissiere/DeepLearningImplementations/blob/master/DenseNet/densenet.py     
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).  The batch size
#' (i.e., number of training images) is not specified a priori. 
#' @param numberOfClassificationLabels Number of segmentation labels.
#' @param depth number of layers---must be equal to 3 * N + 4 where
#' N is an integer (default = 7). 
#' @param numberOfDenseBlocks number of dense blocks to add to the end 
#' (default = 1).
#' @param growthRate number of filters to add for each dense block layer
#' (default = 12).
#' @param dropoutRate = per drop out layer rate (default = 0.2)
#' @param weightDecay = weight decay (default = 1e-4)
#'
#' @return a DenseNet keras model to be used with subsequent fitting
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
#'  outputs <- createDenseNetModel2D( dim( trainingImageArrays[[1]] ), 
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

createDenseNetModel2D <- function( inputImageSize, 
                                   numberOfClassificationLabels = 1000,
                                   numberOfFilters = 16, 
                                   depth = 7,
                                   numberOfDenseBlocks = 1,
                                   growthRate = 12,
                                   dropoutRate = 0.2,
                                   weightDecay = 1e-4
                                 )
{

  if ( ! usePkg( "keras" ) )
    {
    stop( "Please install the keras package." )
    }

  convolutionFactory2D <- function( model, numberOfFilters, kernelSize = c( 3, 3 ), 
                                    dropoutRate = 0.0, weightDecay = 1e-4 )
    {
    model <- model %>% layer_batch_normalization( axis = 1, 
      gamma_regularizer = regularizer_l2( weightDecay ), 
      beta_regularizer = regularizer_l2( weightDecay ) )  
    model <- model %>% layer_activation( activation = 'relu' )
    model <- model %>% layer_conv_2d( filters = numberOfFilters, 
      kernel_size = kernelSize, kernel_initializer = 'he_uniform', padding = 'same', 
      use_bias = FALSE, kernel_regularizer = regularizer_l2( weightDecay ) )
    if( dropoutRate > 0.0 )  
      {
      model <- model %>% layer_dropout( rate = dropoutRate )  
      }
    return( model )
    }

  transition2D <- function( model, numberOfFilters, dropoutRate = 0.0, 
                            weightDecay = 1e-4 )  
    {
    model <- convolutionFactory2D( model, numberOfFilters, kernelSize = c( 1, 1 ),
      dropoutRate = dropoutRate, weightDecay = weightDecay )
    model <- model %>% layer_average_pooling_2d( pool_size = c( 2, 2 ), 
      strides = c( 2, 2 ) )
    return( model )
    }

  createDenseBlocks2D <- function( model, numberOfFilters, depth, growthRate, 
    dropoutRate = 0.0, weightDecay = 1e-4 )
    {
    #  Theano:  [batchSize, channelSize, widthSize, heightSize]
    #  tensorflow:  [batchSize, widthSize, heightSize, channelSize]
    K <- keras::backend()
    concatenationAxis <- 1  
    if( K$image_data_format() == 'channels_last' )
      {
      concatenationAxis <- -1 
      }

    denseBlockLayers <- list( model )
    for( i in 1:depth )  
      {
      model <- convolutionFactory2D( model, numberOfFilters = growthRate, 
        kernelSize = c( 3, 3 ), dropoutRate = dropoutRate, weightDecay = weightDecay )  
      denseBlockLayers[[i+1]] <- model
      model <- layer_concatenate( denseBlockLayers, axis = concatenationAxis )
      numberOfFilters <- numberOfFilters + growthRate
      }
     

    return( list( model = model, numberOfFilters = numberOfFilters ) )  
    }

  if( ( depth - 4 ) %% 3 != 0 )
    {
    stop( "Depth must be equal to 3*N+4 where N is an integer." )  
    }
  numberOfLayers = as.integer( ( depth - 4 ) / 3 )


  inputs <- layer_input( shape = inputImageSize )

  outputs <- inputs %>% layer_conv_2d( filters = numberOfFilters, 
    kernel_size = c( 3, 3 ), kernel_initializer = 'he_uniform', padding = 'same',
    use_bias = FALSE, kernel_regularizer = regularizer_l2( weightDecay ) )
  
  # Add dense blocks

  nFilters <- numberOfFilters

  for( i in 1:( numberOfDenseBlocks - 1 ) )
    {
    denseBlockLayer <- createDenseBlocks2D( outputs, numberOfFilters = nFilters, 
      depth = numberOfLayers, growthRate = growthRate, dropoutRate = dropoutRate,
      weightDecay = weightDecay )
    outputs <- denseBlockLayer$model
    nFilters <- denseBlockLayer$numberOfFilters

    outputs <- transition2D( outputs, numberOfFilters = nFilters, 
      dropoutRate = dropoutRate, weightDecay = weightDecay )
    }

  denseBlockLayer <- createDenseBlocks2D( outputs, numberOfFilters = nFilters, 
    depth = numberOfLayers, growthRate = growthRate, dropoutRate = dropoutRate,
    weightDecay = weightDecay )
  outputs <- denseBlockLayer$model
  nFilters <- denseBlockLayer$numberOfFilters
  
  outputs <- outputs %>% layer_batch_normalization( axis = 1, 
    gamma_regularizer = regularizer_l2( weightDecay ), 
    beta_regularizer = regularizer_l2( weightDecay ) )  
  
  outputs <- outputs %>% layer_activation( activation = 'relu' )
  outputs <- outputs %>% layer_global_average_pooling_2d()
  outputs <- outputs %>% layer_dense( units = numberOfClassificationLabels, 
    activation = 'softmax', kernel_regularizer = regularizer_l2( weightDecay ), 
    bias_regularizer = regularizer_l2( weightDecay ) )

  denseNetModel <- keras_model( inputs = inputs, outputs = outputs )

  return( denseNetModel )
}

#' 3-D implementation of the DenseNet deep learning architecture.
#'
#' Creates a keras model of the DenseNet deep learning architecture for image 
#' recognition based on the paper
#' 
#' G. Huang, Z. Liu, K. Weinberger, and L. van der Maaten. Densely Connected 
#'   Convolutional Networks Networks
#' 
#' available here:
#' 
#'         https://arxiv.org/abs/1608.06993
#'
#' This particular implementation was influenced by the following python 
#' implementation: 
#' 
#'         https://github.com/tdeboissiere/DeepLearningImplementations/blob/master/DenseNet/densenet.py     
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).  The batch size
#' (i.e., number of training images) is not specified a priori. 
#' @param numberOfClassificationLabels Number of segmentation labels.
#' @param depth number of layers---must be equal to 3 * N + 4 where
#' N is an integer (default = 7). 
#' @param numberOfDenseBlocks number of dense blocks to add to the end 
#' (default = 1).
#' @param growthRate number of filters to add for each dense block layer
#' (default = 12).
#' @param dropoutRate = per drop out layer rate (default = 0.2)
#' @param weightDecay = weight decay (default = 1e-4)
#'
#' @return a DenseNet keras model to be used with subsequent fitting
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
#'  outputs <- createDenseNetModel3D( dim( trainingImageArrays[[1]] ), 
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

createDenseNetModel3D <- function( inputImageSize, 
                                   numberOfClassificationLabels = 1000,
                                   numberOfFilters = 16, 
                                   depth = 7,
                                   numberOfDenseBlocks = 1,
                                   growthRate = 12,
                                   dropoutRate = 0.2,
                                   weightDecay = 1e-4
                                 )
{

  if ( ! usePkg( "keras" ) )
    {
    stop( "Please install the keras package." )
    }

  convolutionFactory3D <- function( model, numberOfFilters, kernelSize = c( 3, 3, 3 ), 
                                    dropoutRate = 0.0, weightDecay = 1e-4 )
    {
    model <- model %>% layer_batch_normalization( axis = 1, 
      gamma_regularizer = regularizer_l2( weightDecay ), 
      beta_regularizer = regularizer_l2( weightDecay ) )  
    model <- model %>% layer_activation( activation = 'relu' )
    model <- model %>% layer_conv_3d( filters = numberOfFilters, 
      kernel_size = kernelSize, kernel_initializer = 'he_uniform', padding = 'same', 
      use_bias = FALSE, kernel_regularizer = regularizer_l2( weightDecay ) )
    if( dropoutRate > 0.0 )  
      {
      model <- model %>% layer_dropout( rate = dropoutRate )  
      }
    return( model )
    }

  transition3D <- function( model, numberOfFilters, dropoutRate = 0.0, 
                            weightDecay = 1e-4 )  
    {
    model <- convolutionFactory3D( model, numberOfFilters, kernelSize = c( 1, 1, 1 ),
      dropoutRate = dropoutRate, weightDecay = weightDecay )
    model <- model %>% layer_average_pooling_3d( pool_size = c( 2, 2, 2 ), 
      strides = c( 2, 2, 2 ) )
    return( model )
    }

  createDenseBlocks3D <- function( model, numberOfFilters, depth, growthRate, 
    dropoutRate = 0.0, weightDecay = 1e-4 )
    {
    #  Theano:  [batchSize, channelSize, widthSize, heightSize]
    #  tensorflow:  [batchSize, widthSize, heightSize, channelSize]
    K <- keras::backend()
    concatenationAxis <- 1  
    if( K$image_data_format() == 'channels_last' )
      {
      concatenationAxis <- -1 
      }

    denseBlockLayers <- list( model )
    for( i in 1:depth )  
      {
      model <- convolutionFactory3D( model, numberOfFilters = growthRate, 
        kernelSize = c( 3, 3, 3 ), dropoutRate = dropoutRate, weightDecay = weightDecay )  
      denseBlockLayers[[i+1]] <- model
      model <- layer_concatenate( denseBlockLayers, axis = concatenationAxis )
      numberOfFilters <- numberOfFilters + growthRate
      }
     

    return( list( model = model, numberOfFilters = numberOfFilters ) )  
    }

  if( ( depth - 4 ) %% 3 != 0 )
    {
    stop( "Depth must be equal to 3*N+4 where N is an integer." )  
    }
  numberOfLayers = as.integer( ( depth - 4 ) / 3 )


  inputs <- layer_input( shape = inputImageSize )

  outputs <- inputs %>% layer_conv_3d( filters = numberOfFilters, 
    kernel_size = c( 3, 3, 3 ), kernel_initializer = 'he_uniform', padding = 'same',
    use_bias = FALSE, kernel_regularizer = regularizer_l2( weightDecay ) )
  
  # Add dense blocks

  nFilters <- numberOfFilters

  for( i in 1:( numberOfDenseBlocks - 1 ) )
    {
    denseBlockLayer <- createDenseBlocks3D( outputs, numberOfFilters = nFilters, 
      depth = numberOfLayers, growthRate = growthRate, dropoutRate = dropoutRate,
      weightDecay = weightDecay )
    outputs <- denseBlockLayer$model
    nFilters <- denseBlockLayer$numberOfFilters

    outputs <- transition3D( outputs, numberOfFilters = nFilters, 
      dropoutRate = dropoutRate, weightDecay = weightDecay )
    }

  denseBlockLayer <- createDenseBlocks3D( outputs, numberOfFilters = nFilters, 
    depth = numberOfLayers, growthRate = growthRate, dropoutRate = dropoutRate,
    weightDecay = weightDecay )
  outputs <- denseBlockLayer$model
  nFilters <- denseBlockLayer$numberOfFilters
  
  outputs <- outputs %>% layer_batch_normalization( axis = 1, 
    gamma_regularizer = regularizer_l2( weightDecay ), 
    beta_regularizer = regularizer_l2( weightDecay ) )  
  
  outputs <- outputs %>% layer_activation( activation = 'relu' )
  outputs <- outputs %>% layer_global_average_pooling_3d()
  outputs <- outputs %>% layer_dense( units = numberOfClassificationLabels, 
    activation = 'softmax', kernel_regularizer = regularizer_l2( weightDecay ), 
    bias_regularizer = regularizer_l2( weightDecay ) )

  denseNetModel <- keras_model( inputs = inputs, outputs = outputs )

  return( denseNetModel )
}
