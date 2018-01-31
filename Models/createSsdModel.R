#' 2-D implementation of the SSD deep learning architecture.
#'
#' Creates a keras model of the SSD deep learning architecture for image 
#' recognition based on the paper
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
#'         https://github.com/rykov8/ssd_keras
#'         https://github.com/gsimchoni/ssdkeras
#'
#' @param inputImageSize Used for specifying the input tensor shape.  The
#' shape (or dimension) of that tensor is the image dimensions followed by
#' the number of channels (e.g., red, green, and blue).  The batch size
#' (i.e., number of training images) is not specified a priori. 
#' @param numberOfClassificationLabels Number of classification labels. 
#' Needs to include the background as one of the labels. 
#' @param l2Regularization The L2-regularization rate.  Default = 0.0005.
#' @param minScale The smallest scaling factor for the size of the anchor 
#' boxes as a fraction of the shorter side of the input images.
#' @param maxScale The largest scaling factor for the size of the anchor 
#' boxes as a fraction of the shorter side of the input images. All scaling 
#' factors between the smallest and the largest will be linearly interpolated. 
#' @param aspectRatiosPerLayer A list containing one aspect ratio list for
#' each predictor layer.  The default lists follows the original 
#' implementation.  This variable determines the number of prediction layers.
#' @param variances A list of 4 floats > 0 with scaling factors (actually it's 
#' not factors but divisors to be precise) for the encoded predicted box 
#' coordinates. A variance value of 1.0 would apply no scaling at all to the 
#' predictions, while values in (0,1) upscale the encoded predictions and 
#' values greater than 1.0 downscale the encoded predictions. Defaults to 
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

createSsdModel2D <- function( inputImageSize, 
                              numberOfClassificationLabels,
                              l2Regularization = 0.0005,
                              minScale = 0.1,
                              maxScale = 0.9,
                              aspectRatiosPerLayer = 
                                list( c( 1.0, 2.0, 0.5 ),
                                      c( 1.0, 2.0, 0.5, 3.0, 1.0/3.0 ),
                                      c( 1.0, 2.0, 0.5, 3.0, 1.0/3.0 ),
                                      c( 1.0, 2.0, 0.5, 3.0, 1.0/3.0 ),
                                      c( 1.0, 2.0, 0.5 ),
                                      c( 1.0, 2.0, 0.5 )
                                    ),
                              variances = c( 0.1, 0.1, 0.1, 0.1 )                       
                            )
{

  # Do some initial checking before getting into the graph building

  if ( ! usePkg( "keras" ) )
    {
    stop( "Please install the keras package." )
    }

  if ( ! usePkg( "abind" ) )
    {
    stop( "Please install the abind package." )
    }

  #
  # custom layers:  https://keras.rstudio.com/articles/custom_layers.html
  # https://cran.rstudio.com/web/packages/keras/vignettes/about_keras_layers.html
  #

  # L2 normalization layer described in 
  #
  # Wei Liu, Andrew Rabinovich, and Alexander C. Berg.  ParseNet: Looking Wider 
  #     to See Better.
  #
  # available here:
  #
  #         https://arxiv.org/abs/1506.04579
  # 
  # Input arguments:
  #     * scale:  feature scale (default = 20)
  #
  # Input shape:
  #     Theano:  [batchSize, channelSize, widthSize, heightSize]
  #     tensorflow:  [batchSize, widthSize, heightSize, channelSize]
  #
  # Output shape:
  #     same as input
  #

  L2NormalizationLayer2D <- R6::R6Class( "L2NormalizationLayer2D",
                                    
    inherit = KerasLayer,
    
    public = list(

      scale = NULL,
      
      channelAxis = NULL,

      gamma = NULL, 
      
      initialize = function( scale = 20 ) 
        {
        if( k_image_data_format() == "channels_last" )
          {
          self$channelAxis <- 4  
          } else {
          self$channelAxis <- 2  
          }
        self$scale <- scale  
        },
      
      build = function( input_shape ) 
        {
        self$gamma <- self$add_weight( 
          name = paste0( 'gamma_', self$name ), 
          shape = list( input_shape[[self$channelAxis]] ),
          initializer = initializer_constant( value = self$scale ),
          trainable = TRUE )
        },
      
      call = function( x, mask = NULL ) 
        {
        output <- k_l2_normalize( x, self$channelAxis )
        output <- output * self$gamma
        return( output )
        }
      )
    )

  layer_l2_normalization_2d <- function( object, scale = 20, trainable = TRUE ) {
    create_layer( L2NormalizationLayer2D, object, 
      list( scale = scale, trainable = trainable ) )
  }

  # anchor box layer
  # 
  # Input arguments:
  #     * inputImageSize
  #     * minSize (in pixels) 
  #     * maxSize (in pixels)
  #     * aspectRatios
  #     * variances explained here:
  #            https://github.com/rykov8/ssd_keras/issues/53
  #
  # Input shape:
  #     Theano:  [batchSize, channelSize, widthSize, heightSize]
  #     tensorflow:  [batchSize, widthSize, heightSize, channelSize]
  #
  # Output shape:
  #     5-D tensor [batchSize, widthSize, heightSize, numberOfBoxes, 8]
  #     In the last dimension, the first four correspond to the
  #     xmin, xmax, ymin, ymax of the bounding boxes and the other
  #     four are the variances
  #

  AnchorBoxLayer2D <- R6::R6Class( "AnchorBoxLayer2D" ,
                                    
    inherit = KerasLayer,
    
    public = list(
      
      imageSize = NULL,

      imageSizeAxes = NULL,

      minSize = NULL,

      maxSize = NULL,

      aspectRatios = NULL,

      variances = NULL,
      
      initialize = function( imageSize, minSize, maxSize,
        aspectRatios = c( 0.5, 1.0, 2.0 ), variances = 1.0 )
        {

        #  Theano:  [batchSize, channelSize, widthSize, heightSize]
        #  tensorflow:  [batchSize, widthSize, heightSize, channelSize]

        if( k_image_data_format() == "channels_last" )
          {
          self$imageSizeAxes[1] <- 2  
          self$imageSizeAxes[2] <- 3  
          } else {
          self$imageSizeAxes[1] <- 3  
          self$imageSizeAxes[2] <- 4  
          }
        self$minSize <- minSize
        self$maxSize <- maxSize

        if( is.na( aspectRatios ) )
          {
          self$aspectRatios <- c( 1.0 )
          } else {
          self$aspectRatios <- aspectRatios
          }

        if( length( variances ) == 1 )
          {
          self$variances <- rep( variances, 4 )  
          } else if( length( variances ) == 4 ) {
          self$variances <- variances  
          } else {
          stop( "Error: Length of variances must be 1 or 4." )
          }
        },
            
      call = function( x, mask = NULL ) 
        {
        input_shape <- k_int_shape( x )
        layerSize <- c()
        layerSize[1] <- input_shape[[self$imageSizeAxes[1]]]
        layerSize[2] <- input_shape[[self$imageSizeAxes[2]]]
      
        numberOfBoxes <- length( self$aspectRatios ) * prod( layerSize )

        boxSizes <- list()
        for( i in 1:length( self$aspectRatios ) )
          {
          if( i > 1 && self$aspectRatios[i] == 1 )  
            {
            boxSizes[[i]] <- c( sqrt( self$minSize * self$maxSize ),
                                sqrt( self$minSize * self$maxSize ) )
            } else {
            boxSizes[[i]] <- c( self$minSize * sqrt( self$aspectRatios[i] ),
                                self$minSize / sqrt( self$aspectRatios[i] ) )
            }
          boxSizes[[i]] <- 0.5 * boxSizes[[i]]
          }
        stepSize <- self$imageSize / layerSize
        stepSeq <- list()
        for( i in 1:length( stepSize ) )
          {
          stepSeq[[i]] <- seq( 0.5 * stepSize[i],
            self$imageSize[1] - 0.5 * stepSize[i], length.out = layerSize[i] )
          }

        # Define c( xmin, ymin, xmax, ymax ) of each anchor box
        # We set the initial shape to 
        #    [4, batchSize, widthSize, heightSize, numberOfBoxes]
        # because of the way the array function fills per the leftmost 
        # ordering
        
        anchorBoxesTensor <- 
          array( 0, c( 4, input_shape[[0]], self$imageSize, numberOfBoxes ) )
        for( i in 1:length( self$aspectRatios ) )
          {
          for( j in 1:length( stepSeq[[1]] ) )
            {
            xmin <- stepSeq[[1]][j] - boxSizes[[i]][1]
            xmax <- stepSeq[[1]][j] + boxSizes[[i]][1]
            for( k in 1:length( stepSeq[[2]] ) )
              {
              ymin <- stepSeq[[2]][k] - boxSizes[[i]][2]
              ymax <- stepSeq[[2]][k] + boxSizes[[i]][2]

              anchorBoxCoords <- c( xmin, ymin, xmax, ymax )
              
              anchorBoxesTensor[,, j, k, i] <- array( anchorBoxCoords,
                c( 4, input_shape[[0]], 1, 1, 1 ) )
              count <- count + 1
              }
            }    
          }
        anchorVariancesTensor <- array( variances, c( 4, input_shape[0], 
          self$imageSize, numberOfBoxes ) )  

        permutationOrder <- 
          c( 2:( length( dim( anchorBoxesTensor ) ) - 1 ), 1 )

        anchorBoxesTensor <- 
          aperm( abind( anchorBoxesTensor, anchorVariancesTensor, along = 1 ),          
          perm = permutationOrder )

        return( anchorBoxesTensor )  
        },

      compute_output_shape = function( input_shape ) 
        {
        layerSize <- c()
        layerSize[1] <- input_shape[[self$imageSizeChannels[1]]]
        layerSize[2] <- input_shape[[self$imageSizeChannels[2]]]
        numberOfBoxes <- length( self$aspectRatios ) * prod( layerSize )
        return ( c( input_shape[0], numberOfBoxes, 8 ) )
        }
      )
    )

  layer_anchor_box_2d <- function( object ) {
    create_layer( AnchorBoxLayer2D, object )
  }

  inputs <- layer_input( shape = inputImageSize )

  filterSizes <- c( 64, 128, 256, 512, 1024 ) 

  numberOfPredictorLayers <- length( aspectRatiosPerLayer )

  numberOfBoxesPerLayer <- rep( 0, numberOfPredictorLayers )
  for( i in 1:numberOfPredictorLayers )
    {
    numberOfBoxesPerLayer[i] <- length( aspectRatiosPerLayer[[i]] )  
    }

  scales <- seq( from = minScale, to = maxScale, 
    length.out = numberOfPredictorLayers + 1 )

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
  numberOfCoordinates <- 2^2

  # Initial convolutions 1-4

  outputs <- inputs
  for( i in 1:4 )
    {
    outputs <- outputs %>% layer_conv_2d( filters = filterSizes[i], 
      kernel_size = c( 3, 3 ), activation = 'relu', padding = 'same', 
      kernel_initializer = initializer_he_normal(), 
      kernel_regularizer = regularizer_l2( l2Regularization ) ) 

    outputs <- outputs %>% layer_conv_2d( filters = filterSizes[i], 
      kernel_size = c( 3, 3 ), activation = 'relu', padding = 'same', 
      kernel_initializer = initializer_he_normal(), 
      kernel_regularizer = regularizer_l2( l2Regularization ) ) 

    if( i > 2 ) 
      {
      outputs <- outputs %>% layer_conv_2d( filters = filterSizes[i], 
        kernel_size = c( 3, 3 ), activation = 'relu', padding = 'same', 
        kernel_initializer = initializer_he_normal(), 
        kernel_regularizer = regularizer_l2( l2Regularization ) ) 

      if( i == 4 )
        {
        l2NormalizedOutputs <- outputs %>% 
          layer_l2_normalization_2d( scale = 20 )
        }
      }

    outputs <- outputs %>% layer_max_pooling_2d( pool_size = c( 2, 2 ), 
      strides = c( 2, 2 ), padding = 'same' )
    }

  boxClasses[[1]] <- l2NormalizedOutputs %>% layer_conv_2d( 
    filters = numberOfBoxesPerLayer[1] * numberOfClassificationLabels, 
    kernel_size = c( 3, 3 ),
    padding = 'same', kernel_initializer = initializer_he_normal(),
    kernel_regularizer = regularizer_l2( l2Regularization ) )

  boxLocations[[1]] <- l2NormalizedOutputs %>% layer_conv_2d( 
    filters = numberOfBoxesPerLayer[1] * numberOfCoordinates,
    kernel_size = c( 3, 3 ),
    padding = 'same', kernel_initializer = initializer_he_normal(),
    kernel_regularizer = regularizer_l2( l2Regularization ) )

  # Conv5

  outputs <- outputs %>% layer_conv_2d( filters = filterSizes[4], 
    kernel_size = c( 3, 3 ), activation = 'relu', padding = 'same', 
    kernel_initializer = initializer_he_normal(), 
    kernel_regularizer = regularizer_l2( l2Regularization ) ) 

  outputs <- outputs %>% layer_conv_2d( filters = filterSizes[i], 
    kernel_size = c( 3, 3 ), activation = 'relu', padding = 'same', 
    kernel_initializer = initializer_he_normal(), 
    kernel_regularizer = regularizer_l2( l2Regularization ) ) 

  outputs <- outputs %>% layer_conv_2d( filters = filterSizes[4], 
    kernel_size = c( 3, 3 ), activation = 'relu', padding = 'same', 
    kernel_initializer = initializer_he_normal(), 
    kernel_regularizer = regularizer_l2( l2Regularization ) ) 

  outputs <- outputs %>% layer_max_pooling_2d( pool_size = c( 3, 3 ), 
    strides = c( 1, 1 ), padding = 'same' )

  # fc6

  outputs <- outputs %>% layer_conv_2d( filters = filterSizes[5],
    kernel_size = c( 3, 3 ), dilation_rate = c( 6, 6 ), 
    activation = 'relu', padding = 'same', 
    kernel_initializer = initializer_he_normal(), 
    kernel_regularizer = regularizer_l2( l2Regularization ) ) 

  # fc7

  outputs <- outputs %>% layer_conv_2d( filters = filterSizes[5],
    kernel_size = c( 1, 1 ), 
    activation = 'relu', padding = 'same', 
    kernel_initializer = initializer_he_normal(), 
    kernel_regularizer = regularizer_l2( l2Regularization ) ) 

  boxClasses[[2]] <- outputs %>% layer_conv_2d( 
    filters = numberOfBoxesPerLayer[2] * numberOfClassificationLabels, 
    kernel_size = c( 3, 3 ),
    padding = 'same', kernel_initializer = initializer_he_normal(),
    kernel_regularizer = regularizer_l2( l2Regularization ) )

  boxLocations[[2]] <- outputs %>% layer_conv_2d( 
    filters = numberOfBoxesPerLayer[2] * numberOfCoordinates, 
    kernel_size = c( 3, 3 ),
    padding = 'same', kernel_initializer = initializer_he_normal(),
    kernel_regularizer = regularizer_l2( l2Regularization ) )

  # Conv6

  outputs <- outputs %>% layer_conv_2d( filters = filterSizes[3],
    kernel_size = c( 1, 1 ), 
    activation = 'relu', padding = 'same', 
    kernel_initializer = initializer_he_normal(), 
    kernel_regularizer = regularizer_l2( l2Regularization ) ) 

  outputs <- outputs %>% layer_zero_padding_2d( padding = c( 1, 1 ) )  

  outputs <- outputs %>% layer_conv_2d( filters = filterSizes[4],
    kernel_size = c( 3, 3 ), strides = c( 2, 2 ), 
    activation = 'relu', padding = 'valid', 
    kernel_initializer = initializer_he_normal(), 
    kernel_regularizer = regularizer_l2( l2Regularization ) ) 

  boxClasses[[3]] <- outputs %>% layer_conv_2d( 
    filters = numberOfBoxesPerLayer[3] * numberOfClassificationLabels, 
    kernel_size = c( 3, 3 ),
    padding = 'same', kernel_initializer = initializer_he_normal(),
    kernel_regularizer = regularizer_l2( l2Regularization ) )

  boxLocations[[3]] <- outputs %>% layer_conv_2d( 
    filters = numberOfBoxesPerLayer[3] * numberOfCoordinates, 
    kernel_size = c( 3, 3 ),
    padding = 'same', kernel_initializer = initializer_he_normal(),
    kernel_regularizer = regularizer_l2( l2Regularization ) )

  # Conv7

  outputs <- outputs %>% layer_conv_2d( filters = filterSizes[2],
    kernel_size = c( 1, 1 ), 
    activation = 'relu', padding = 'same', 
    kernel_initializer = initializer_he_normal(), 
    kernel_regularizer = regularizer_l2( l2Regularization ) ) 

  outputs <- outputs %>% layer_zero_padding_2d( padding = c( 1, 1 ) )  

  outputs <- outputs %>% layer_conv_2d( filters = filterSizes[3],
    kernel_size = c( 3, 3 ), strides = c( 2, 2 ), 
    activation = 'relu', padding = 'valid', 
    kernel_initializer = initializer_he_normal(), 
    kernel_regularizer = regularizer_l2( l2Regularization ) ) 

  boxClasses[[4]] <- outputs %>% layer_conv_2d( 
    filters = numberOfBoxesPerLayer[4] * numberOfClassificationLabels, 
    kernel_size = c( 3, 3 ),
    padding = 'same', kernel_initializer = initializer_he_normal(),
    kernel_regularizer = regularizer_l2( l2Regularization ) )

  boxLocations[[4]] <- outputs %>% layer_conv_2d( 
    filters = numberOfBoxesPerLayer[4] * numberOfCoordinates, 
    kernel_size = c( 3, 3 ),
    padding = 'same', kernel_initializer = initializer_he_normal(),
    kernel_regularizer = regularizer_l2( l2Regularization ) )

  # Conv8

  outputs <- outputs %>% layer_conv_2d( filters = filterSizes[2],
    kernel_size = c( 1, 1 ), 
    activation = 'relu', padding = 'same', 
    kernel_initializer = initializer_he_normal(), 
    kernel_regularizer = regularizer_l2( l2Regularization ) ) 

  outputs <- outputs %>% layer_conv_2d( filters = filterSizes[3],
    kernel_size = c( 3, 3 ), strides = c( 1, 1 ), 
    activation = 'relu', padding = 'valid', 
    kernel_initializer = initializer_he_normal(), 
    kernel_regularizer = regularizer_l2( l2Regularization ) ) 

  boxClasses[[5]] <- outputs %>% layer_conv_2d( 
    filters = numberOfBoxesPerLayer[5] * numberOfClassificationLabels, 
    kernel_size = c( 3, 3 ),
    padding = 'same', kernel_initializer = initializer_he_normal(),
    kernel_regularizer = regularizer_l2( l2Regularization ) )

  boxLocations[[5]] <- outputs %>% layer_conv_2d( 
    filters = numberOfBoxesPerLayer[5] * numberOfCoordinates, 
    kernel_size = c( 3, 3 ),
    padding = 'same', kernel_initializer = initializer_he_normal(),
    kernel_regularizer = regularizer_l2( l2Regularization ) )

  # Conv9

  outputs <- outputs %>% layer_conv_2d( filters = filterSizes[2],
    kernel_size = c( 1, 1 ), 
    activation = 'relu', padding = 'same', 
    kernel_initializer = initializer_he_normal(), 
    kernel_regularizer = regularizer_l2( l2Regularization ) ) 

  outputs <- outputs %>% layer_conv_2d( filters = filterSizes[3],
    kernel_size = c( 3, 3 ), strides = c( 1, 1 ), 
    activation = 'relu', padding = 'valid', 
    kernel_initializer = initializer_he_normal(), 
    kernel_regularizer = regularizer_l2( l2Regularization ) ) 

  boxClasses[[6]] <- outputs %>% layer_conv_2d( 
    filters = numberOfBoxesPerLayer[6] * numberOfClassificationLabels, kernel_size = c( 3, 3 ),
    padding = 'same', kernel_initializer = initializer_he_normal(),
    kernel_regularizer = regularizer_l2( l2Regularization ) )

  boxLocations[[6]] <- outputs %>% layer_conv_2d( 
    filters = numberOfBoxesPerLayer[6] * numberOfCoordinates, kernel_size = c( 3, 3 ),
    padding = 'same', kernel_initializer = initializer_he_normal(),
    kernel_regularizer = regularizer_l2( l2Regularization ) )

  # Generate the anchor boxes.  Output shape of anchor boxes =
  #   ``( batch, height, width, numberOfBoxes, 8 )``
  anchorBoxes <- list()

  shortImageSize <- min( inputImageSize[1:2] )

  for( i in 1:length( boxLocations ) )
    {
    anchorBoxes[[i]] <- boxLocations[[i]] %>% 
      layer_anchor_box_2d( inputImageSize, 
        minSize = ( scales[i] * shortImageSize ), 
        maxSize = ( scales[i+1] * shortImageSize ),
        aspectRatios = aspectRatiosPerLayer[[i]], variances = variances )
    }

  # Reshape the box confidence values, box locations, and 
  boxClassesReshaped <- list()
  boxLocationsReshaped <- list()
  for( i in 1:length( boxClasses ) )
    {
    boxClassesReshaped[[i]] <- boxClasses[[i]] %>% 
      layer_reshape( target_shape = c( -1, numberOfClassificationLabels ) )
    boxLocationsReshaped[[i]] <- boxLocations[[i]] %>% 
      layer_reshape( target_shape = c( -1, numberOfCoordinates ) )  
    anchorBoxesReshaped[[i]] <- anchorBoxes[[i]] %>% 
      layer_reshape( target_shape = c( -1, 8 ) )
    }  
  
  # Concatenate the predictions from the different layers

  outputClasses <- 
    layer_concatenate( boxClassesReshaped, axis = 1 )
  outputLocations <- layer_concatenate( boxLocationsReshaped, axis = 1 )
  outputAnchorBoxes <- layer_concatenate( anchorBoxesReshaped, axis = 1 )

  confidenceActivation <- outputClasses %>% 
    layer_activation( activation = "softmax" )

  predictions <- layer_concatenate( list( 
    confidenceActivation, outputLocations, outputAnchorBoxes ), axis = 2 )

  ssdModel <- keras_model( inputs = inputs, outputs = predictions )

  return( ssdModel )
}
