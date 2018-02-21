#' @export

ssdImageBatchGenerator <- R6::R6Class( "SsdImageBatchGenerator",

  public = list( 
    
    imageList = NULL,

    labels = NULL,

    initialize = function( imageList = NULL, labels = NULL )
      {
      if( !usePkg( "magick" ) )
        {
        stop( "Please install the magick package." )
        }

      if( !usePkg( "abind" ) )
        {
        stop( "Please install the abind package." )
        }

      if( !is.null( imageList ) )
        {
        self$imageList <- imageList
        } else {
        self$imageList <- list()
        }

      if( !is.null( labels ) )
        {
        self$labels <- labels
        } else {
        self$labels <- list()
        }
      },

    generate = function( batchSize = 32L, anchorBoxes = NULL, 
      variances = rep( 1.0, 4 ), equalize = FALSE, brightness = NULL, 
      flipHorizontally = NULL, translate = NULL, scale = FALSE )    
      {

      # shuffle the data
      sampleIndices <- sample( length( self$imageList ) )
      self$imageList <- self$imageList[sampleIndices]
      self$labels <- self$labels[sampleIndices]

      currentPassCount <- 1L

      function() 
        {
        # Shuffle the data after each complete pass 

        if( currentPassCount >= length( self$imageList ) )
          {
          sampleIndices <- sample( length( self$imageList ) )
          self$imageList <- self$imageList[sampleIndices]
          self$labels <- self$labels[sampleIndices]

          currentPassCount <- 1L
          }

        batchIndices <- currentPassCount:min( 
          ( currentPassCount + batchSize - 1L ), length( self$imageList ) )
        batchImages <- self$imageList[batchIndices]
        batchX <- abind( batchImages, along = 4 )

        dimBatchX <- dim( batchX )
        nDimBatchX <- length( dimBatchX )

        imageSize <- dimBatchX[1:(nDimBatchX - 1)]
        batchSize <- dimBatchX[nDimBatchX]

        batchX <- aperm( batchX, c( nDimBatchX, 1:( nDimBatchX - 1 ) ) )
        batchY <- self$labels[batchIndices]

        currentPassCount <- currentPassCount + batchSize

        # Boxes format is numberOfBoxes x ( xmin, xmax, ymin, ymax )
        translateBoxes <- function( boxes, shift )
          {
          boxes[, 1:2] <- boxes[, 1:2] + shift[1]
          boxes[, 3:4] <- boxes[, 3:4] + shift[2]
          }

        # Boxes format is numberOfBoxes x ( xmin, xmax, ymin, ymax )
        flipHorizontalBoxes <- function( boxes, imageWidth )
          {
          boxes[, 1:2] <- imageWidth - boxes[, 2:1] + imageWidth
          }

        # Boxes format is numberOfBoxes x ( xmin, xmax, ymin, ymax )
        scaleBoxes <- function( boxes, scaleFactor )
          {
          boxes <- boxes * scaleFactor
          }

        for( i in seq_len( batchSize ) )
          {
          if( equalize ) 
            {
            tempX <- image_read( batchX[i,,,] / 255 ) %>%
              image_equalize() %>%  
              .[[1]] %>% as.numeric()
            batchX[i,,,] <- tempX[,, 1:3] * 255
            }

          if( !is.null( brightness ) )  
            {
            brightValue <- as.integer( 
              runif( 1, min = brightness[1], max = brightness[2] ) * 100 )

            tempX <- image_read( batchX[i,,,] / 255 ) %>%
              image_modulate( brightValue ) %>%
              .[[1]] %>% as.numeric()
            batchX[i,,,] <- tempX[,, 1:3] * 255
            }

          if( !is.null( flipHorizontally ) && runif( 1 ) < flipHorizontally )  
            {
            tempX <- image_read( batchX[i,,,] / 255 ) %>%
              image_flop() %>%  
              .[[1]] %>% as.numeric()
            batchX[i,,,] <- tempX[,, 1:3] * 255
            batchY[[i]][,2:5] <- 
              flipHorizontalBoxes( batchY[[i]][,2:5], imageSize[1] )
            }
          
          if( !is.null( translate ) && runif( 1 ) < translate[[3]] )  
            {
            blankImage <- image_read( array( runif( 1 ), imageSize ) ) 
            shift <- c( 0, 0 )  
            shift[1] <- sample( c( -1, 1 ), 1 ) * 
              as.integer( runif( 1, translate[[1]][1], translate[[1]][2] ) )
            shift[2] <- sample( c( -1, 1 ), 1 ) * 
              as.integer( runif( 1, translate[[2]][1], translate[[2]][2] ) )
            shiftStringX <- ifelse( shift[1] > 0, 
              paste0( "+", shift[1] ), as.character( shift[1] ) )  
            shiftStringY <- ifelse( shift[2] > 0, 
              paste0( "+", shift[2] ), as.character( shift[2] ) )   
            offsetString <- paste0( shiftStringX, shiftStringY )

            tempImage <- image_read( batchX[i,,,] / 255 )
            tempX <- image_composite( blankImage, tempImage, 
              offset = offsetString ) %>%  
              .[[1]] %>% as.numeric()

            batchX[i,,,] <- tempX[,, 1:3] * 255
            batchY[[i]][,2:5] <- translateBoxes( batchY[[i]][,2:5], shift )
            }

          if( !is.null( scale ) && runif( 1 ) < scale[3] )  
            {
            blankImage <- image_read( array( runif( 1 ), imageSize ) ) 
            scaleFactor <- round( runif( 1, scale[1], scale[2] ), 2 )
            scaleString <- paste0( scaleFactor * 100, "%" )
            scaleGeometryString <- paste0( scaleString, "x", scaleString )

            tempImage <- image_read( batchX[i,,,] / 255 ) %>%
              image_scale( scaleGeometryString )
            tempX <- image_composite( blankImage, tempImage ) %>%
              .[[1]] %>% as.numeric()

            batchX[i,,,] <- tempX[,, 1:3] * 255
            batchY[[i]][, 2:5] <- scaleBoxes( batchY[[i]][, 2:5], scaleFactor )
            }
          }

        encodedBatchY <- encodeY( batchY, anchorBoxes, imageSize, variances )  
        return( list( batchX, encodedBatchY ) )        
        }   
      }
    )
  )