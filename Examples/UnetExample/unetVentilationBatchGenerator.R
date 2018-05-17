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

      },

    generate = function( batchSize = 32L, resampledImageSize = c( 128, 128 ) )    
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

        channelSize <- length( batchImages[[1]] )

        batchX <- array( data = 0, dim = c( batchSize, resampledImageSize, channelSize ) )
        batchY <- array( data = 0, dim = c( batchSize, resampledImageSize ) )

        currentPassCount <<- currentPassCount + batchSize

        for( i in seq_len( batchSize ) )
          {
          subjectBatchImages <- batchImages[[i]]  

          referenceX <- antsImageRead( batchReferenceImages[[i]][1], dimension = 2 )
          referenceXfrm <- batchReferenceTransforms[[i]]

          sourceXfrm <- batchTransforms[[i]]

          boolInvert <- c( TRUE, FALSE, FALSE, FALSE )
          transforms <- c( referenceXfrm$invtransforms[1], 
            referenceXfrm$invtransforms[2], sourceXfrm$fwdtransforms[1],
            sourceXfrm$fwdtransforms[2] )

          sourceY <- antsImageRead( batchSegmentations[[i]], dimension = 2 )

          warpedImageY <- antsApplyTransforms( referenceX, sourceY, 
            interpolator = "genericLabel", transformlist = transforms,
            whichtoinvert = boolInvert  )

          if( any( dim( warpedImageY ) != resampledImageSize ) )
            {
            warpedArrayY <- as.array( resampleImage( warpedImageY, 
              resampledImageSize, useVoxels = TRUE, interpType = 1 ) )
            } else {
            warpedArrayY <- as.array( warpedImageY )
            }

          # antsImageWrite( as.antsImage( warpedArrayY ), "~/Desktop/arrayY.nii.gz" )
          batchY[i,,] <- warpedArrayY

          # Randomly "flip a coin" to see if we perform histogram matching.

          doPerformHistogramMatching <- sample( c( TRUE, FALSE ), size = 1 )
          doPerformHistogramMatching <- FALSE

          for( j in seq_len( channelSize ) )
            {  
            sourceX <- antsImageRead( subjectBatchImages[j], dimension = 2 )

            warpedImageX <- antsApplyTransforms( referenceX, sourceX, 
              interpolator = "linear", transformlist = transforms,
              whichtoinvert = boolInvert )

            if( doPerformHistogramMatching )
              {
              warpedImageX <- histogramMatchImage( warpedImageX, 
                antsImageRead( batchReferenceImages[[i]][j], dimension = 2 ),
                numberOfHistogramBins = 64, numberOfMatchPoints = 16 )
              }

            if( any( dim( warpedImageX ) != resampledImageSize ) )
              {
              warpedArrayX <- as.array( resampleImage( warpedImageX, 
                resampledImageSize, useVoxels = TRUE, interpType = 0 ) )
              } else {
              warpedArrayX <- as.array( warpedImageX )
              }

            warpedArrayX <- ( warpedArrayX - mean( warpedArrayX ) ) / 
              sd( warpedArrayX )  

            # antsImageWrite( as.antsImage( warpedArrayX ), "~/Desktop/arrayX.nii.gz" )
            # readline( prompt = "Press [enter] to continue\n" )
            batchX[i,,,j] <- warpedArrayX
            }

          }
        segmentationLabels <- sort( unique( as.vector( batchY ) ) )
        encodedBatchY <- encodeUnet( batchY, segmentationLabels ) 

        return( list( batchX, encodedBatchY ) )        
        }   
      }
    )
  )