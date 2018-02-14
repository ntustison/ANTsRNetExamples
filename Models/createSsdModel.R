#' Loss function for the SSD deep learning architecture.
#'
#' Creates an R6 class object for use with the SSD deep learning architecture
#' based on the paper
#' 
#' W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C-Y. Fu, A. Berg. 
#'     SSD: Single Shot MultiBox Detector.
#' 
#' available here:
#' 
#'         https://arxiv.org/abs/1512.02325
#'
#' This particular implementation was heavily influenced by the following 
#' python and R implementations: 
#' 
#'         https://github.com/rykov8/ssd_keras
#'         https://github.com/gsimchoni/ssdkeras/blob/master/R/ssd_loss.R
#'
#' @param backgroundRatio The maximum ratio of background to foreround
#' for weighting in the loss function.  Is rounded to the nearest integer.
#' Default is '3'.
#' @param minNumberOfBackgroundBoxes The minimum number of background boxes
#' to use in loss computation *per batch*.  Should reflect a value in 
#' proportion to the batch size.  Default is 0.
#' @param alpha Weighting factor for the localization loss in total loss 
#' computation.
#'
#' @return an SSD loss function
#' @author Tustison NJ
#' @examples
#'
#' \dontrun{ 
#' 
#' library( keras )
#' 
#' }

lossSsd <- R6::R6Class( "LossSSD",

  public = list( 
      
    backgroundRatio = 3L, 
    
    minNumberOfBackgroundBoxes = 0L, 
    
    alpha = 1.0,
                         
    numberOfClassificationLabels = NULL,

    # Can we generalize beyond tensorflow?
    tf = tensorflow::tf,

    initialize = function( backgroundRatio = 3L, 
      minNumberOfBackgroundBoxes = 0L, alpha = 1.0, 
      numberOfClassificationLabels = NULL ) 
      {
      self$backgroundRatio <- self$tf$constant( backgroundRatio )
      self$minNumberOfBackgroundBoxes <- 
        self$tf$constant( minNumberOfBackgroundBoxes )
      self$alpha <- self$tf$constant( alpha )
      self$numberOfClassificationLabels <- 
        as.integer( numberOfClassificationLabels )
      },
      
    smooth_l1_loss = function( y_true, y_pred ) 
      {
      y_true <- self$tf$cast( y_true, dtype = "float32" )
      absoluteLoss <- self$tf$abs( y_true - y_pred )
      squareLoss <- 0.5 * ( y_true - y_pred )^2
      l1Loss <- self$tf$where( self$tf$less( absoluteLoss, 1.0 ), 
        squareLoss, absoluteLoss - 0.5 )
      return( self$tf$reduce_sum( l1Loss, axis = -1L ) )
      },

    log_loss = function( y_true, y_pred ) 
      {
      y_true <- self$tf$cast( y_true, dtype = "float32" )
      y_pred <- self$tf$maximum( y_pred, 1e-15 )
      logLoss <- 
        -self$tf$reduce_sum( y_true * self$tf$log( y_pred ), axis = -1L )
      return( logLoss )
      },

    compute_loss = function( y_true, y_pred ) 
      {
      y_true$set_shape( y_pred$get_shape() )
      batchSize <- self$tf$shape( y_pred )[1] 
      numberOfBoxesPerCell <- self$tf$shape( y_pred )[2] 

      classificationLoss <- self$tf$to_float( self$log_loss( 
         y_true[,, 1:self$numberOfClassificationLabels], 
         y_pred[,, 1:self$numberOfClassificationLabels] ) ) 
      localizationLoss <- self$tf$to_float( self$smooth_l1_loss( 
        y_true[,, ( self$numberOfClassificationLabels + 1 ):
                  ( self$numberOfClassificationLabels + 4 )], 
        y_pred[,, ( self$numberOfClassificationLabels + 1 ):
                  ( self$numberOfClassificationLabels + 4 )] ) )

      backgroundBoxes <- y_true[,, 1] 
      foregroundBoxes <- self$tf$to_float( self$tf$reduce_max( 
        y_true[,, 2:self$numberOfClassificationLabels], axis = -1L ) ) 

      numberOfForegroundBoxes <- self$tf$reduce_sum( foregroundBoxes )

      foregroundClassLoss <- self$tf$reduce_sum( 
        classificationLoss * foregroundBoxes, axis = -1L )

      backgroundClassLossAll <- classificationLoss * backgroundBoxes
      nonZeroIndices <- 
        self$tf$count_nonzero( backgroundClassLossAll, dtype = self$tf$int32 )

      numberOfBackgroundBoxesToKeep <- self$tf$minimum( self$tf$maximum( 
        self$backgroundRatio * self$tf$to_int32( numberOfForegroundBoxes ), 
        self$minNumberOfBackgroundBoxes ), nonZeroIndices )

      f1 = function() 
        {
        return( self$tf$zeros( list( batchSize ) ) )
        }

      f2 = function() 
        {
        backgroundClassLossAll1d <- 
          self$tf$reshape( backgroundClassLossAll, list( -1L ) )
        topK <- self$tf$nn$top_k( 
          backgroundClassLossAll1d, numberOfBackgroundBoxesToKeep, FALSE )
        values <- topK$values
        indices <- topK$indices

        backgroundBoxesToKeep <- self$tf$scatter_nd( 
          self$tf$expand_dims( indices, axis = 1L ), 
          updates = self$tf$ones_like( indices, dtype = self$tf$int32 ), 
          shape = self$tf$shape( backgroundClassLossAll1d ) ) 
        backgroundBoxesToKeep <- self$tf$to_float( 
          self$tf$reshape( backgroundBoxesToKeep, 
          list( batchSize, numberOfBoxesPerCell ) ) )

        return( self$tf$reduce_sum( 
          classificationLoss * backgroundBoxesToKeep, axis = -1L ) )
        }

      backgroundClassLoss <- self$tf$cond( self$tf$equal( 
        nonZeroIndices, self$tf$constant( 0L ) ), f1, f2 )

      classLoss <- foregroundClassLoss + backgroundClassLoss

      localizationLoss <- 
        self$tf$reduce_sum( localizationLoss * foregroundBoxes, axis = -1L )

      totalLoss <- ( classLoss + self$alpha * localizationLoss ) / 
        self$tf$maximum( 1.0, numberOfForegroundBoxes ) 

      return( totalLoss )
      }
    )
  )

#' 2-D Jaccard similarity between two sets of boxes.
#'
#' Function for determinining the Jaccard or iou (intersection over union) 
#' similarity measure between two sets of boxes.
#'
#' @param boxes1 A 2-D array where each row corresponds to a single box 
#' consisting of the format (xmin,xmax,ymin,ymax)
#' @param boxes2 A 2-D array where each row corresponds to a single box 
#' consisting of the format (xmin,xmax,ymin,ymax)
#'
#' @return an SSD loss function
#' @author Tustison NJ
#' @examples
#'
#' \dontrun{ 
#' 
#' library( keras )
#' 
#' }

jaccardSimilarity2D <- function( boxes1, boxes2 )
  {
  np <- reticulate::import( "numpy" )  

  if( is.null( dim( boxes1 ) ) )
    {
    boxes1 <- np$expand_dims( boxes1, axis = 0L )  
    }
  if( is.null( dim( boxes2 ) ) )
    {
    boxes2 <- np$expand_dims( boxes2, axis = 0L )  
    }

  intersection <- np$maximum( 0, np$minimum( boxes1[, 2], boxes2[, 2] ) - 
                                 np$maximum( boxes1[, 1], boxes2[, 1] ) ) * 
                  np$maximum( 0, np$minimum( boxes1[, 4], boxes2[, 4] ) - 
                                 np$maximum( boxes1[, 3], boxes2[, 3] ) )

  union <- ( boxes1[, 2] - boxes1[, 1] ) * ( boxes1[, 4] - boxes1[, 3] ) +
    ( boxes2[, 2] - boxes2[, 1] ) * ( boxes2[, 4] - boxes2[, 3] ) - 
    intersection
  return( intersection / union )
  }

#' Encoding function for Y_train
#'
#' Function for translating the min/max ground truth box coordinates to 
#' something expected by the SSD network.  This is a SSD-specific analog
#' for keras::to_categorical().  For each image in the batch, we compare
#' the ground truth boxes for that image with all the anchor boxes.  If 
#' the overlap measure exceeds a specific threshold, we write the ground
#' truth box coordinates and class to the specific position of the matched
#' anchor box.  Note that the background class will be assigned to all the 
#' anchor boxes for which there was no match with any ground truth box.
#' However, an exception to this are the anchor boxes whose overlap measure
#' is higher that the specified negative threshold. 
#'
#' This particular implementation was heavily influenced by the following 
#' python and R implementations: 
#' 
#'         https://github.com/pierluigiferrari/ssd_keras  
#'         https://github.com/rykov8/ssd_keras
#'         https://github.com/gsimchoni/ssdkeras
#'
#' @param groundTruthLabels A list of length `batchSize` that contains one 
#' 2-D array per image.  Each 2-D array has k rows where each row corresponds
#' to a single box consisting of the format 
#' 
#'          classId,xmin,xmax,ymin,max
#'
#' Note that `classId` must be greater than 0 since 0 is reserved for the 
#' background label.
#' @param anchorBoxes a list of 2-D arrays where each element comprises the
#' anchor boxes for a specific aspect ratios layer.  The row of each 2-D array
#' comprises a single box specified in the form
#'
#'          xmin, xmax, ymin, ymax
#'
#' @param variances A list of 4 floats > 0 with scaling factors (actually it's 
#' not factors but divisors to be precise) for the encoded predicted box 
#' coordinates. A variance value of 1.0 would apply no scaling at all to the 
#' predictions, while values in (0,1) upscale the encoded predictions and 
#' values greater than 1.0 downscale the encoded predictions. These are the same
#' variances used to construct the model.
#' @param foregroundThreshold float between 0 and 1 determining the min threshold 
#' for matching an anchor box with a ground truth box and, thus, labeling an anchor 
#' box as a non-background class.  If an anchor box exceeds the ``backgroundThreshold`` 
#' but does not meet the foregroundThreshold for a ground truth box, then it is ignored 
#' during training.  Default = 0.5.
#' @param backgroundThreshold float between 0 and 1 determining the max threshold 
#' for labeling an anchor box as `background`.  If an anchor box exceeds the 
#' ``backgroundThreshold`` but does not meet the foregroundThreshold for a ground
#' truth box, then it is ignored during training.  Default = 0.3.
#'
#' @return a 3-D array of shape 
#'      
#'         `(batchSize, numberOfBoxes, numberOfClasses + 4 + 4 + 4)`
#'
#' where the additional 4's along the third dimension correspond to 
#' the box coordinates (xmin, xmax, ymin, ymax), dummy variables, and
#' the variances.
#'
#' @author Tustison NJ
#' @examples
#'
#' \dontrun{ 
#' 
#' library( keras )
#' 
#' }

encodeY <- function( groundTruthLabels, anchorBoxes, variances,
  foregroundThreshold = 0.5, backgroundThreshold = 0.3 )
  {
  np <- reticulate::import( "numpy" )  

  batchSize <- length( groundTruthLabels )
  classIds <- c()
  for( i in 1:batchSize )
    {
    classIds <- append( classIds, groundTruthLabels[[i]][, 1] )
    }
  classIds <- sort( unique( c( 0, classIds ) ) )
  numberOfClassificationLabels <- length( classIds )

  numberOfBoxes <- 0L
  for( i in 1:length( anchorBoxes ) )
    {
    numberOfBoxes <- numberOfBoxes + nrow( anchorBoxes[[i]] )  
    }  

  anchorBoxesList <- list()
  for( i in 1:length( anchorBoxes ) )
    {
    anchorBoxExpanded <- np$expand_dims( anchorBoxes[[i]], axis = 0L )      
    anchorBoxExpanded <- np$tile( anchorBoxes[[i]], c( batchSize, 1L, 1L ) )
    anchorBoxesList[[i]] <- anchorBoxExpanded
    }
  boxesTensor <- np$concatenate( anchorBoxesList, axis = 1L )
  classesTensor <- np$zeros( reticulate::tuple( 
    batchSize, numberOfBoxes, numberOfClassificationLabels ) )
  variancesTensor <- np$zeros_like( boxesTensor ) + variances

  # ``boxesTensor`` is concatenated the second time as a space filler
  yEncodedTemplate <- np$concatenate( reticulate::tuple( 
    classesTensor, boxesTensor, boxesTensor, variancesTensor ), axis = 2L )
  yEncoded = np$copy( yEncodedTemplate )

  # We now fill in ``yEncoded``

  # identity matrix used for one-hot encoding
  classEye <- np$eye( numberOfClassificationLabels )

  boxIndices <- 
    ( numberOfClassificationLabels + 1 ):( numberOfClassificationLabels + 4 )
  classIndices <- 1:(numberOfClassificationLabels + 4 )
 
  for( i in 1:batchSize )
    { 
    availableBoxes <- np$ones( numberOfBoxes )
    backgroundBoxes <- np$ones( numberOfBoxes )

    for( j in 1:nrow( groundTruthLabels[[i]] ) )
      {
      groundTruthBox <- groundTruthLabels[[i]][j,]
      groundTruthCoords <- as.numeric( groundTruthBox[-1] )
      groundTruthLabel <- as.integer( groundTruthBox[1] )

      similarities <- jaccardSimilarity2D( 
        yEncodedTemplate[i, , boxIndices], groundTruthCoords )
      
      # check to see which boxes exceed the background threshold and are no 
      # longer potential background boxes.  Also, clear out those background 
      # boxes from the ``similarities`` list.
      backgroundBoxes[similarities >= backgroundThreshold] <- 0
      similarities <- similarities * availableBoxes

      availableAndThreshold <- np$copy( similarities )
      availableAndThreshold[availableAndThreshold < foregroundThreshold] <- 0

      nonZeroIndices <- np$nonzero( availableAndThreshold )[[1]] + 1
      if( length( nonZeroIndices ) > 0 )
        {
        yEncoded[i, nonZeroIndices, classIndices] <- rep( 
          np$concatenate( reticulate::tuple( classEye[groundTruthLabel + 1,],
            groundTruthCoords ), axis = 0L ), each = length( nonZeroIndices ) )
        availableBoxes[nonZeroIndices] <- 0
        } else {
        bestMatchIndex <- np$argmax( similarities ) + 1
        yEncoded[i, bestMatchIndex, classIndices] <- 
          np$concatenate( reticulate::tuple( classEye[groundTruthLabel + 1,], 
            groundTruthCoords ), axis = 0L )
        availableBoxes[bestMatchIndex] <- 0
        backgroundBoxes[bestMatchIndex] <- 0
        }
      }
    # Set the remaining background indices to the background class  
    backgroundClassIndices = np$nonzero( backgroundBoxes )[[1]] + 1
    yEncoded[i, backgroundClassIndices, 1] <- 1
    }

  return( yEncoded )
  }

#' Decoding function for Y_train
#'
#' Function for translating the predictions from the SSD model output to
#' boxes, (xmin, xmax, ymin, ymax), for subsequent usage.
#'
#' This particular implementation was heavily influenced by the following 
#' python and R implementations: 
#' 
#'         https://github.com/pierluigiferrari/ssd_keras  
#'         https://github.com/rykov8/ssd_keras
#'         https://github.com/gsimchoni/ssdkeras
#'
#' @param yPredicted The predicted output produced by the SSD model expected to
#' be an array of shape
#'
#'         `(batchSize, numberOfBoxes, numberOfClasses + 4 + 4 + 4)`
#'
#' where the additional 4's along the third dimension correspond to the box 
#' coordinates (xmin, xmax, ymin, ymax), dummy variables, and the variances.
#' @param confidenceThreshold  Float between 0 and 1.  The minimum 
#' classification value required for a given box to be considered a "positive
#' prediction."  A lower value will result in better recall while a higher 
#' value yields higher precision results.  Default = 0.5.
#' @param overlapThreshold  'NULL' or a float between 0 and 1.  If 'NULL' then
#' no non-maximum suppression will be performed.  Otherwise, a greedy non-
#' maximal suppression is performed following confidence thresholding.  In
#' other words all boxes with Jaccard similarities > ``overlapThreshold`` will
#' be removed from the set of predictions.   Default = 0.45.
#'
#' @return a list of length ``batchSize`` where each element comprises a 2-D
#' array where each row describes a single box using the following six elements
#' 
#'          `(classId, confidenceValue, xmin, xmax, ymin, ymax)`
#'
#' @author Tustison NJ
#' @examples
#'
#' \dontrun{ 
#' 
#' library( keras )
#' 
#' }

decodeY <- function( yPredicted, confidenceThreshold = 0.5, 
  overlapThreshold = 0.45 )
  {
  np <- reticulate::import( "numpy" )  

  greedyNonMaximalSuppression <- function( predictions,
    overlapThreshold = 0.45 )
    { 
    predictionsLeft <- np$copy( predictions )

    index <- 1
    maximumBoxList <- list()
    while( !is.null( dim( predictionsLeft ) ) 
      && dim( predictionsLeft )[1] > 0 )
      {
      maximumIndex <- np$argmax( predictionsLeft[, 2] ) + 1L
      maximumBox <- np$copy( predictionsLeft[maximumIndex, ] )
      maximumBoxList[[index]] <- maximumBox
      index <- index + 1
      yPredictedLeft <- 
        np$delete( predictionsLeft, maximumIndex - 1L, axis = 0L )
      if( is.null( dim( predictionsLeft ) ) )
        {
        break  
        }
      similarities <- jaccardSimilarity2D( 
        predictionsLeft[, 3:6], array( maximumBox[3:6], c( 1, 4 ) ) )
      predictionsLeft <- predictionsLeft[similarities <= overlapThreshold, ]  
      }
    return( do.call( rbind, maximumBoxList ) )  
    }
  
  numberOfClassificationLabels <- dim( yPredicted )[3] - 12L

  yPredictedConverted <- np$copy( yPredicted[,, 
    ( numberOfClassificationLabels - 1 ):( numberOfClassificationLabels + 4 ), 
    drop = FALSE] )

  # store class ID  
  yPredictedConverted[,, 1] <- 
    np$argmax( yPredicted[,, 1:numberOfClassificationLabels], axis = -1L )

  # store confidence values  
  yPredictedConverted[,, 2] <- 
    np$amax( yPredicted[,, 1:numberOfClassificationLabels], axis = -1L )

  yDecoded <- list()
  for( i in seq_len( dim( yPredictedConverted )[1] ) )
    {
    ySingle <- yPredictedConverted[i,,]  

    boxes <- ySingle[unlist( np$nonzero( ySingle[, 1] ) ) + 1,, drop = FALSE]
    boxes <- boxes[boxes[, 2] >= confidenceThreshold,, drop = FALSE]

    if( !is.null( overlapThreshold ) )
      {
      boxes <- greedyNonMaximalSuppression( boxes, overlapThreshold )  
      }
    if( is.null( boxes ) )
      {  
      yDecoded[[i]] <- matrix(, nrow = 0, ncol = 6 )
      } else {
      yDecoded[[i]] <- boxes  
      }  
    }
  return( yDecoded )
  }

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
#' factors between the smallest and the largest are linearly interpolated. 
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
        },

      compute_output_shape = function( input_shape ) 
        {
        return( reticulate::tuple( input_shape ) )
        }
      )
    )

  layer_l2_normalization_2d <- function( object, scale = 20 ) {
    create_layer( L2NormalizationLayer2D, object, 
      list( scale = scale ) )
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

      minSize = NULL,

      maxSize = NULL,

      aspectRatios = NULL,

      variances = NULL,

      imageSizeAxes = NULL,

      channelAxis = NULL, 

      numberOfBoxes = NULL,

      anchorBoxesArray = NULL,
      
      initialize = function( imageSize, minSize, maxSize,
        aspectRatios = c( 0.5, 1.0, 2.0 ), variances = 1.0 )
        {

        #  Theano:  [batchSize, channelSize, widthSize, heightSize]
        #  tensorflow:  [batchSize, widthSize, heightSize, channelSize]

        if( k_image_data_format() == "channels_last" )
          {
          self$imageSizeAxes[1] <- 2  
          self$imageSizeAxes[2] <- 3  
          self$channelAxis <- 4
          } else {
          self$imageSizeAxes[1] <- 3  
          self$imageSizeAxes[2] <- 4  
          self$channelAxis <- 2
          }
        self$minSize <- minSize
        self$maxSize <- maxSize

        self$imageSize <- imageSize

        if( is.null( aspectRatios ) )
          {
          self$aspectRatios <- c( 1.0 )
          } else {
          self$aspectRatios <- aspectRatios
          }
        self$numberOfBoxes <- length( aspectRatios )

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
        np <- reticulate::import( "numpy" )

        input_shape <- k_int_shape( x )
        layerSize <- c()
        layerSize[1] <- input_shape[[self$imageSizeAxes[1]]]
        layerSize[2] <- input_shape[[self$imageSizeAxes[2]]]
      
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

        # Define c( xmin, xmax, ymin, ymax ) of each anchor box

        coordCount <- 1
        self$anchorBoxesArray <- array( NA, dim = c( 0, 4 ) )

        anchorBoxesTuple <- np$zeros( reticulate::tuple( 
          layerSize[1], layerSize[2], self$numberOfBoxes, 4L ) )
        anchorVariancesTuple <- np$zeros( reticulate::tuple( 
          layerSize[1], layerSize[2], self$numberOfBoxes, 4L ) )  
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
              
              anchorBoxCoords <- c( xmin, xmax, ymin, ymax )
              
              if( coordCount == 1 )
                {
                self$anchorBoxesArray <- anchorBoxCoords
                } else {
                self$anchorBoxesArray <- 
                  rbind( self$anchorBoxesArray, anchorBoxCoords )
                }
              coordCount <- coordCount + 1

              anchorBoxesTuple[j, k, i,] <- anchorBoxCoords
              anchorVariancesTuple[j, k, i,] <- variances
              }
            }    
          }

        anchorBoxesTensor <- np$concatenate( reticulate::tuple( 
          anchorBoxesTuple, anchorVariancesTuple ), axis = -1L )
        anchorBoxesTensor <- np$expand_dims( anchorBoxesTensor, axis = 0L )  

        anchorBoxesTensor <- k_constant( anchorBoxesTensor, dtype = 'float32' )
#        anchorBoxesTensor <- k_tile( anchorBoxesTensor, 
#          c( k_shape( x )[1], 1L, 1L, 1L, 1L ) )
        anchorBoxesTensor <- keras::backend()$tile( anchorBoxesTensor, 
          c( k_shape( x )[1], 1L, 1L, 1L, 1L ) )

        return( anchorBoxesTensor )  
        },

      compute_output_shape = function( input_shape ) 
        {
        layerSize <- c()
        layerSize[1] <- input_shape[[self$imageSizeAxes[1]]]
        layerSize[2] <- input_shape[[self$imageSizeAxes[2]]]

        return( reticulate::tuple( input_shape[[1]], layerSize[1], 
          layerSize[2], self$numberOfBoxes, 8L ) )
        }
      )
    )

  layer_anchor_box_2d <- function( object, 
    imageSize, minSize, maxSize, aspectRatios, variances ) {
    create_layer( AnchorBoxLayer2D, object, 
      list( imageSize = imageSize, minSize = minSize, maxSize = maxSize, 
            aspectRatios = aspectRatios, variances = variances )
      )
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

  imageDimension <- 2
  numberOfCoordinates <- 2^imageDimension

  # Initial convolutions 1-4

  outputs <- inputs

  numberOfLayers <- 4
  for( i in 1:numberOfLayers )
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

      if( i == numberOfLayers )
        {
        l2NormalizedOutputs <- outputs %>% 
          layer_l2_normalization_2d( scale = 20 )

        boxClasses[[1]] <- outputs %>% layer_conv_2d( 
          filters = numberOfBoxesPerLayer[1] * numberOfClassificationLabels, 
          kernel_size = c( 3, 3 ),
          padding = 'same', kernel_initializer = initializer_he_normal(),
          kernel_regularizer = regularizer_l2( l2Regularization ) )

        boxLocations[[1]] <- outputs %>% layer_conv_2d( 
          filters = numberOfBoxesPerLayer[1] * numberOfClassificationLabels,
          kernel_size = c( 3, 3 ),
          padding = 'same', kernel_initializer = initializer_he_normal(),
          kernel_regularizer = regularizer_l2( l2Regularization ) )

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

  outputs <- outputs %>% layer_conv_2d( filters = filterSizes[4], 
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

  outputs <- outputs %>% layer_zero_padding_2d( 
    padding = list( c( 1, 1 ), c( 1, 1 ) ) )  

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

  outputs <- outputs %>% layer_zero_padding_2d( 
    padding = list( c( 1, 1 ), c( 1, 1 ) ) ) 

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
    filters = numberOfBoxesPerLayer[6] * numberOfClassificationLabels, 
    kernel_size = c( 3, 3 ), padding = 'same', 
    kernel_initializer = initializer_he_normal(),
    kernel_regularizer = regularizer_l2( l2Regularization ) )

  boxLocations[[6]] <- outputs %>% layer_conv_2d( 
    filters = numberOfBoxesPerLayer[6] * numberOfCoordinates, 
    kernel_size = c( 3, 3 ), padding = 'same', 
    kernel_initializer = initializer_he_normal(),
    kernel_regularizer = regularizer_l2( l2Regularization ) )

  # Generate the anchor boxes.  Output shape of anchor boxes =
  #   ``( batch, height, width, numberOfBoxes, 8 )``
  anchorBoxes <- list()
  anchorBoxLayers <- list()
  predictorSizes <- list()

  imageSize <- inputImageSize[1:imageDimension]
  shortImageSize <- min( imageSize )

  for( i in 1:length( boxLocations ) )
    {
    anchorBoxLayer <- layer_anchor_box_2d( imageSize = imageSize, 
      minSize = ( scales[i] * shortImageSize ), 
      maxSize = ( scales[i+1] * shortImageSize ),
      aspectRatios = aspectRatiosPerLayer[[i]], variances = variances )
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
      target_shape = c( inputShape[[2]] * inputShape[[3]] * numberOfBoxes, 
        numberOfClassificationLabels ) )

    # reshape ``( batch, height, width, numberOfBoxes * 4 )``
    #   to `( batch, height * width * numberOfBoxes, 4 )`
    boxLocationsReshaped[[i]] <- boxLocations[[i]] %>% layer_reshape( 
      target_shape = c( inputShape[[2]] * inputShape[[3]] * numberOfBoxes, 4 ) )

    # reshape ``( batch, height, width, numberOfBoxes * 8 )``
    #   to `( batch, height * width * numberOfBoxes, 8 )`
    anchorBoxLayersReshaped[[i]] <- anchorBoxLayers[[i]] %>% layer_reshape( 
      target_shape = c( inputShape[[2]] * inputShape[[3]] * numberOfBoxes, 8 ) )
    }  
  
  # Concatenate the predictions from the different layers

  outputClasses <- layer_concatenate( boxClassesReshaped, axis = 1 )
  outputLocations <- layer_concatenate( boxLocationsReshaped, axis = 1 )
  outputAnchorBoxes <- layer_concatenate( anchorBoxLayersReshaped, axis = 1 )

  confidenceActivation <- outputClasses %>% 
    layer_activation( activation = "softmax" )

  predictions <- layer_concatenate( list( 
    confidenceActivation, outputLocations, outputAnchorBoxes ), axis = 2 )

  ssdModel <- keras_model( inputs = inputs, outputs = predictions )

  return( list( ssdModel = ssdModel, anchorBoxes = anchorBoxes ) )
}
