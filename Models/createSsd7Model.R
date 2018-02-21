#' 2-D implementation of the SSD 7 deep learning architecture.
#'
#' Creates a keras model of the SSD 7 deep learning architecture for 
#' object detection based on the paper
#' 
#' W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C-Y. Fu, A. Berg. 
#'     SSD: Single Shot MultiBox Detector.
#' 
#' available here:
#' 
#'         https://arxiv.org/abs/1512.02325
#'
#' This particular implementation was influenced by the following python 
#' and R implementations: 
#' 
#'         https://github.com/pierluigiferrari/ssd_keras     
#'         https://github.com/gsimchoni/ssdkeras
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).  The batch size
#' (i.e., number of training images) is not specified a priori. 
#' @param numberOfClassificationLabels Number of classification labels. 
#' Needs to include the background as one of the labels. 
#' @param minScale The smallest scaling factor for the size of the anchor 
#' boxes as a fraction of the shorter side of the input images.
#' @param maxScale The largest scaling factor for the size of the anchor 
#' boxes as a fraction of the shorter side of the input images. All scaling 
#' factors between the smallest and the largest are linearly interpolated. 
#' @param aspectRatiosPerLayer A list containing one aspect ratio list for
#' each predictor layer.  The default lists follows the original 
#' implementation.  This variable determines the number of prediction layers.
#' @param variances A list of 4 floats > 0 with scaling factors for the encoded 
#' predicted box coordinates. A variance value of 1.0 would apply no scaling at 
#' all to the predictions, while values in (0,1) upscale the encoded predictions 
#' and values greater than 1.0 downscale the encoded predictions. Defaults to 
#' c( 0.1, 0.1, 0.1, 0.1 ).
#'
#' @return an SSD keras model
#' @author Tustison NJ
#' @examples
#'
#' \dontrun{ 
#' 
#' library( keras )
#' 
#' }

createSsd7Model2D <- function( inputImageSize, 
                              numberOfClassificationLabels,
                              minScale = 0.08,
                              maxScale = 0.96,
                              aspectRatiosPerLayer = 
                                list( c( 0.5, 1.0, 2.0 ),
                                      c( 0.5, 1.0, 2.0 ),
                                      c( 0.5, 1.0, 2.0 ),
                                      c( 0.5, 1.0, 2.0 )
                                    ),
                              variances = rep( 1.0, 4 )
                            )
{
  if( !usePkg( "keras" ) )
    {
    stop( "Please install the keras package." )
    }

  filterSizes <- c( 32, 48, 64, 64, 48, 48, 32 ) 

  numberOfPredictorLayers <- length( aspectRatiosPerLayer )

  numberOfBoxesPerLayer <- rep( 0, numberOfPredictorLayers )
  for( i in 1:numberOfPredictorLayers )
    {
    numberOfBoxesPerLayer[i] <- length( aspectRatiosPerLayer[[i]] )  
    }

  scales <- seq( from = minScale, to = maxScale, 
    length.out = numberOfPredictorLayers + 1 )

  imageDimension <- 2
  numberOfCoordinates <- 2^imageDimension

  # For each of the ``numberOfClassificationLabels``, we predict confidence 
  # values for each box.  This translates into each confidence predictor 
  # having a depth of  ``numberOfBoxesPerLayer`` * 
  # ``numberOfClassificationLabels``.
  boxClasses <- list()

  # For each box we need to predict the 2^imageDimension coordinates.  The 
  # output shape of these localization layers is:
  # ( batchSize, imageHeight, imageWidth, 
  #      numberOfBoxesPerLayer * 2^imageDimension )
  boxLocations <- list()

  # Initial convolutions 1-4

  inputs <- layer_input( shape = inputImageSize )

  outputs <- inputs

  for( i in 1:length( filterSizes ) )
    {
    kernelSize <- c( 5, 5 )
    if( i > 1 )
      {
      kernelSize <- c( 3, 3 )  
      }

    outputs <- outputs %>% layer_conv_2d( filters = filterSizes[i], 
      kernel_size = kernelSize, strides = c( 1, 1 ), 
      padding = 'same' )

    outputs <- outputs %>% 
      layer_batch_normalization( axis = 3, momentum = 0.99 )
     
    outputs <- outputs %>% layer_activation_elu()

    if( i < length( filterSizes ) )
      {
      outputs <- outputs %>% layer_max_pooling_2d( pool_size = c( 2, 2 ) )
      }

    if( i >= 4 )  
      {
      index <- i - 3  
      boxClasses[[index]] <- outputs %>% layer_conv_2d( 
        filters = numberOfBoxesPerLayer[index] * numberOfClassificationLabels, 
        kernel_size = c( 3, 3 ), strides = c( 1, 1 ),
        padding = 'valid' )

      boxLocations[[index]] <- outputs %>% layer_conv_2d( 
        filters = numberOfBoxesPerLayer[index] * numberOfCoordinates, 
        kernel_size = c( 3, 3 ), strides = c( 1, 1 ),
        padding = 'valid' )
      }

    }

  # Generate the anchor boxes.  Output shape of anchor boxes =
  #   ``( batch, height, width, numberOfBoxes, 8 )``
  anchorBoxes <- list()
  anchorBoxLayers <- list()
  predictorSizes <- list()

  imageSize <- inputImageSize[1:imageDimension]
  shortImageSize <- min( imageSize )

  layerNames <- paste0( c( "conv4_3_norm", "fc7", "conv6_2", "conv7_2", 
    "conv8_2", "conv9_2" ), "_mbox" )

  for( i in 1:length( boxLocations ) )
    {
    anchorBoxLayer <- layer_anchor_box_2d( imageSize = imageSize, 
      minSize = ( scales[i] * shortImageSize ), 
      maxSize = ( scales[i+1] * shortImageSize ),
      aspectRatios = aspectRatiosPerLayer[[i]], variances = variances, 
      name = paste0( layerNames[i], "_priorbox" ) )
    anchorBoxLayers[[i]] <- boxLocations[[i]] %>% anchorBoxLayer

    # We calculate the anchor box values again to return as output for 
    # encoding Y_train.  I'm guessing there's a better way to do this 
    # but it's the cleanest I've found.
    anchorBoxGenerator <- AnchorBoxLayer2D$new( imageSize = imageSize,
      minSize = ( scales[i] * shortImageSize ), 
      maxSize = ( scales[i+1] * shortImageSize ),
      aspectRatios = aspectRatiosPerLayer[[i]], variances = variances )
    anchorBoxGenerator$call( boxLocations[[i]] )  
    anchorBoxes[[i]] <- anchorBoxGenerator$anchorBoxesArray
    }

  # Reshape the box confidence values, box locations, and 
  boxClassesReshaped <- list()
  boxLocationsReshaped <- list()
  anchorBoxLayersReshaped <- list()
  for( i in 1:length( boxClasses ) )
    {
    # reshape ``( batch, height, width, numberOfBoxes * numberOfClasses )``
    #   to ``(batch, height * width * numberOfBoxes, numberOfClasses )``
    inputShape <- k_int_shape( boxClasses[[i]] )
    numberOfBoxes <- 
      as.integer( inputShape[[4]] / numberOfClassificationLabels )

    boxClassesReshaped[[i]] <- boxClasses[[i]] %>% layer_reshape( 
      target_shape = c( -1, numberOfClassificationLabels ) )

    # reshape ``( batch, height, width, numberOfBoxes * 4 )``
    #   to `( batch, height * width * numberOfBoxes, 4 )`
    boxLocationsReshaped[[i]] <- boxLocations[[i]] %>% layer_reshape( 
      target_shape = c( -1, 4 ) )

    # reshape ``( batch, height, width, numberOfBoxes * 8 )``
    #   to `( batch, height * width * numberOfBoxes, 8 )`
    anchorBoxLayersReshaped[[i]] <- anchorBoxLayers[[i]] %>% layer_reshape( 
      target_shape = c( -1, 8 ) )
    }  
  
  # Concatenate the predictions from the different layers

  outputClasses <- layer_concatenate( boxClassesReshaped, axis = 1 )
  outputLocations <- layer_concatenate( boxLocationsReshaped, axis = 1 )
  outputAnchorBoxes <- layer_concatenate( anchorBoxLayersReshaped, axis = 1 )

  confidenceActivation <- outputClasses %>% 
    layer_activation( activation = "softmax", name = "mbox_conf_softmax" )

  predictions <- layer_concatenate( list( confidenceActivation, 
    outputLocations, outputAnchorBoxes ), axis = 2 )

  ssdModel <- keras_model( inputs = inputs, outputs = predictions )

  return( list( ssdModel = ssdModel, anchorBoxes = anchorBoxes ) )
}
