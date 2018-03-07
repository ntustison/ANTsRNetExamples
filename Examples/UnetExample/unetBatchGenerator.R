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

      if( !usePkg( "abind" ) )
        {
        stop( "Please install the abind package." )
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

        currentPassCount <- currentPassCount + batchSize

        for( i in seq_len( batchSize ) )
          {
          sourceX <- batchImages[[i]]
          sourceY <- batchSegmentations[[i]]
          sourceXfrm <- batchTransforms[[i]]

          randomIndex <- sample.int( length( self$imageList ), size = 1 )
          referenceX <- self$imageList[[randomIndex]]
          referenceY <- self$segmentationList[[randomIndex]]
          referenceXfrm <- self$transformList[[randomIndex]]

          # transformList <- list()
          # transformList[[1]] <- list( fwdtransforms = reg$fwdtransforms, invtransforms = invtransforms )

          boolInvert <- c( TRUE, FALSE, FALSE, FALSE )
          transforms <- list( referenceXfrm$invtransforms[[1]], 
            referenceXfrm$invtransforms[[2]], sourceXfrm$fwdtransforms[[1]],
            sourceXfrm$fwdtransforms[[2]] )

          warpedX <- antsApplyTransforms( referenceX, sourceX, 
            interpolator = "linear", transformList = transforms,
            whichtoinverse = boolInvert )          
          warpedY <- antsApplyTransforms( referenceX, sourceX, 
            interpolator = "genericLabel", transformList = transforms,
            whichtoinvert = boolInvert )

          batchX[i,,,1] <- as.array( warpedX )[,]
          batchY[i,,,1] <- as.array( warpedY )[,]
          }
        return( list( batchX, batchY ) )        
        }   
      }
    )
  )