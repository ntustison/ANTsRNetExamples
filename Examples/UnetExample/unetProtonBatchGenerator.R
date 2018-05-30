#' @export

unetImageBatchGenerator <- R6::R6Class( "UnetImageBatchGenerator",

  public = list( 
    
    imageList = NULL,

    segmentationList = NULL,

    transformList = NULL,

    referenceImageList = NULL,

    referenceTransformList = NULL,

    pairwiseIndices = NULL,

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

      if( is.null( referenceImageList ) || 
        is.null( referenceTransformList ) )
        {
        self$referenceImageList <- imageList
        self$referenceTransformList <- transformList
        } else {
        self$referenceImageList <- referenceImageList
        self$referenceTransformList <- referenceTransformList
        }

      self$pairwiseIndices <- expand.grid( source = 1:length( self$imageList ), 
        reference = 1:length( self$referenceImageList ) )  

      # shuffle the pairs
      self$pairwiseIndices <- 
        self$pairwiseIndices[sample.int( nrow( self$pairwiseIndices ) ),]
      },

    generate = function( batchSize = 32L )    
      {

      # shuffle the source data
      sampleIndices <- sample( length( self$imageList ) )
      self$imageList <- self$imageList[sampleIndices]
      self$segmentationList <- self$segmentationList[sampleIndices]
      self$transformList <- self$transformList[sampleIndices]
     
      # shuffle the reference data
      sampleIndices <- sample( length( self$referenceImageList ) )
      self$referenceImageList <- self$referenceImageList[sampleIndices]
      self$referenceTransformList <- self$referenceTransformList[sampleIndices]

      currentPassCount <- 1L

      function() 
        {
        # Shuffle the data after each complete pass 

        if( currentPassCount >= nrow( self$pairwiseIndices ) )
          {
          # shuffle the source data
          sampleIndices <- sample( length( self$imageList ) )
          self$imageList <- self$imageList[sampleIndices]
          self$segmentationList <- self$segmentationList[sampleIndices]
          self$transformList <- self$transformList[sampleIndices]

          # shuffle the reference data
          sampleIndices <- sample( length( self$referenceImageList ) )
          self$referenceImageList <- self$referenceImageList[sampleIndices]
          self$referenceTransformList <- self$referenceTransformList[sampleIndices]

          currentPassCount <- 1L
          }

        rowIndices <- currentPassCount + 0:( batchSize - 1L )

        outOfBoundsIndices <- which( rowIndices > nrow( self$pairwiseIndices ) )
        while( length( outOfBoundsIndices ) > 0 )
          {
          rowIndices[outOfBoundsIndices] <- rowIndices[outOfBoundsIndices] - 
            nrow( self$pairwiseIndices )
          outOfBoundsIndices <- which( rowIndices > nrow( self$pairwiseIndices ) )
          }
        batchIndices <- self$pairwiseIndices[rowIndices,]

        batchImages <- self$imageList[batchIndices$source]
        batchSegmentations <- self$segmentationList[batchIndices$source]
        batchTransforms <- self$transformList[batchIndices$source]

        batchReferenceImages <- self$referenceImageList[batchIndices$reference]
        batchReferenceTransforms <- self$referenceTransformList[batchIndices$reference]

        imageSize <- dim( batchImages[[1]] )

        batchX <- array( data = 0, dim = c( batchSize, imageSize, 1 ) )
        batchY <- array( data = 0, dim = c( batchSize, imageSize ) )

        currentPassCount <<- currentPassCount + batchSize

        for( i in seq_len( batchSize ) )
          {
          sourceX <- batchImages[[i]] 
          sourceY <- batchSegmentations[[i]]
          sourceXfrm <- batchTransforms[[i]]

          referenceX <- batchReferenceImages[[i]]
          referenceXfrm <- batchReferenceTransforms[[i]]

          boolInvert <- c( TRUE, FALSE, FALSE, FALSE )
          transforms <- c( referenceXfrm$invtransforms[1], 
            referenceXfrm$invtransforms[2], sourceXfrm$fwdtransforms[1],
            sourceXfrm$fwdtransforms[2] )

          warpedImageX <- antsApplyTransforms( referenceX, sourceX, 
            interpolator = "linear", transformlist = transforms,
            whichtoinvert = boolInvert )          
          warpedImageY <- antsApplyTransforms( referenceX, sourceY, 
            interpolator = "genericLabel", transformlist = transforms,
            whichtoinvert = boolInvert )

          doPerformHistogramMatching <- sample( c( TRUE, FALSE ), size = 1 )
          if( doPerformHistogramMatching )
            {
            warpedImageX <- histogramMatchImage( warpedImageX, referenceX,
              numberOfHistogramBins = 64, numberOfMatchPoints = 16 )
            }

          warpedArrayX <- as.array( warpedImageX )
          warpedArrayY <- as.array( warpedImageY )

          warpedArrayX <- ( warpedArrayX - mean( warpedArrayX ) ) / sd( warpedArrayX )
          # warpedArrayX <- ( warpedArrayX - min( warpedArrayX ) ) / 
          #   ( max( warpedArrayX ) - min( warpedArrayX ) )

          batchX[i,,, 1] <- warpedArrayX
          batchY[i,,] <- warpedArrayY
          }

        segmentationLabels <- sort( unique( as.vector( batchY ) ) )

        encodedBatchY <- encodeUnet( batchY, segmentationLabels ) 

        return( list( batchX, encodedBatchY ) )        
        }   
      }
    )
  )