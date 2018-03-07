#' @export

unetImageBatchGenerator <- R6::R6Class( "UnetImageBatchGenerator",

  public = list( 
    
    imageList = NULL,

    segmentationList = NULL,

    transformList = NULL,

    initialize = function( imageList = NULL, segmentationList = NULL, 
      transformList = NULL )
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
        batchY <- array( data = 0, dim = c( batchSize, imageSize, 1 ) )

        currentPassCount <- currentPassCount + batchSize

        for( i in seq_len( batchSize ) )
          {
          sourceX <- batchImages[[i]]
          sourceY <- batchSegmentations[[i]]
          sourceXfrm <- batchTransforms[[i]]

          randomIndex <- sample.int( length( self$imageList ), size = 1 )
          referenceX <- self$imageList[[randomIndex]]
          referenceXfrm <- self$transformList[[randomIndex]]

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
          batchY[i,,,1] <- as.array( warpedY )
          }

        # Now encode batchY

        segmentationLabels <- sort( unique( as.vector( batchY ) ) )
        numberOfLabels <- length( segmentationLabels )

        encodedBatchY <- batchY
        encodedBatchY[which( batchY == 0 )] <- 1
        encodedBatchY[which( batchY != 0 )] <- 0

        for( i in 2:numberOfLabels )
          {
          labelY <- batchY
          labelY[which( batchY == segmentationLabels[i] )] <- 1
          labelY[which( batchY != segmentationLabels[i] )] <- 0

          encodedBatchY <- abind( encodedBatchY, labelY, along = 4 )
          }
        
        return( list( batchX, encodedBatchY ) )        
        }   
      }
    )
  )