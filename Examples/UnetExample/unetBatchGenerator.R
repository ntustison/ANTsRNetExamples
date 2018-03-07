#' @export

unetImageBatchGenerator <- R6::R6Class( "UnetImageBatchGenerator",

  public = list( 
    
    imageList = NULL,

    segmentationList = NULL,

    transformList = NULL,

    referenceImageList = NULL,

    referenceTransformList = NULL,

    initialize = function( imageList = NULL, segmentationList = NULL, 
      transformList = NULL, referenceImageList = NULL, 
      referenceTransformList = NULL )
      {
      if( !usePkg( "ANTsR" ) )
        {
        stop( "Please install the ANTsR package." )
        }

      if( !is.null( imageList ) )
        {
        self$imageList <- imageList
        } else {
        stop( "Input images must be specified." )
        }

      if( !is.null( segmentationList ) )
        {
        self$segmentationList <- segmentationList
        } else {
        stop( "Input segmentation images must be specified." )
        }

      if( !is.null( transformList ) )
        {
        self$transformList <- transformList
        } else {
        stop( "Input transforms must be specified." )
        }

      self$referenceImageList = referenceImageList
      self$referenceTransformList = referenceTransformList
      },

    generate = function( batchSize = 32L )    
      {
      # shuffle the data
      sampleIndices <- sample( length( self$imageList ) )
      self$imageList <- self$imageList[sampleIndices]
      self$segmentationList <- self$segmentationList[sampleIndices]
      self$transformList <- self$transformList[sampleIndices]

      currentPassCount <- 1L

      function() 
        {
        # Shuffle the data after each complete pass 

        if( currentPassCount >= length( self$imageList ) )
          {
          sampleIndices <- sample( length( self$imageList ) )
          self$imageList <- self$imageList[sampleIndices]
          self$segmentationList <- self$segmentationList[sampleIndices]
          self$transformList <- self$transformList[sampleIndices]

          currentPassCount <- 1L
          }

        batchIndices <- currentPassCount:min( 
          ( currentPassCount + batchSize - 1L ), length( self$imageList ) )

        batchImages <- self$imageList[batchIndices]
        batchSegmentations <- self$segmentationList[batchIndices]
        batchTransforms <- self$transformList[batchIndices]

        batchSize <- length( batchImages )
        imageSize <- dim( batchImages[[1]] )

        batchX <- array( data = 0, dim = c( batchSize, imageSize, 1 ) )
        batchY <- array( data = 0, dim = c( batchSize, imageSize ) )

        currentPassCount <- currentPassCount + batchSize

        for( i in seq_len( batchSize ) )
          {
          sourceX <- batchImages[[i]]
          sourceY <- batchSegmentations[[i]]
          sourceXfrm <- batchTransforms[[i]]
          
          if( is.null( self$referenceImageList ) || 
            is.null( self$referenceTransformList ) )
            {
            randomIndex <- sample.int( length( self$imageList ), size = 1 )
            referenceX <- self$imageList[[randomIndex]]
            referenceXfrm <- self$transformList[[randomIndex]]
            } else {
            randomIndex <- sample.int( length( self$referenceImageList ), size = 1 )
            referenceX <- self$referenceImageList[[randomIndex]]
            referenceXfrm <- self$referenceTransformList[[randomIndex]]
            }

          boolInvert <- c( TRUE, FALSE, FALSE, FALSE )
          transforms <- c( referenceXfrm$invtransforms[1], 
            referenceXfrm$invtransforms[2], sourceXfrm$fwdtransforms[1],
            sourceXfrm$fwdtransforms[2] )

          warpedX <- antsApplyTransforms( referenceX, sourceX, 
            interpolator = "linear", transformlist = transforms,
            whichtoinvert = boolInvert )          
          warpedY <- antsApplyTransforms( referenceX, sourceY, 
            interpolator = "genericLabel", transformlist = transforms,
            whichtoinvert = boolInvert )

          batchX[i,,,1] <- as.array( warpedX )
          batchY[i,,] <- as.array( warpedY )
          }

        segmentationLabels <- sort( unique( as.vector( batchY ) ) )

        encodedBatchY <- encodeY( batchY, segmentationLabels ) 

        return( list( batchX, encodedBatchY ) )        
        }   
      }
    )
  )