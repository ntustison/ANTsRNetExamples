library( ANTsR )
library( xml2 )
library( tidyverse )
library( stringr )
library( keras )
library( ggplot2 )
library( jpeg )

visuallyInspectEachImage <- FALSE

numberOfTrainingData <- 946
numberOfTestingData <- 54
testingImageFiles <- rep( NA, numberOfTestingData )

baseDirectory <- './'
dataDirectory <- paste0( baseDirectory, './lfw_faces_tagged/' )
imageDirectory <- paste0( dataDirectory, 'Images/' )
annotationsDirectory <- paste0( dataDirectory, 'Annotations/' )
dataFile <- paste0( dataDirectory, 'data.csv' )

modelDirectory <- paste0( baseDirectory, '../../Models/' )

source( paste0( modelDirectory, 'createSsd7Model.R' ) )
source( paste0( modelDirectory, 'ssdUtilities.R' ) )

parseXML <- function( xml, labels ) {
  
  frame <- xml %>%
    xml_find_first("//filename") %>%
    xml_text()
  
  classes <- xml %>%
    xml_find_all("//object") %>%
    xml_find_all(".//name") %>%
    xml_text() %>%
    factor( levels = labels ) %>%
    as.integer() %>%
    as_tibble() %>%
    magrittr::set_colnames( "class_id" )
  
  bndbx <- xml %>%
    xml_find_all("//bndbox") %>%
    xml_children() %>%
    xml_integer() %>%
    split( rep( 1:dim( classes )[1], each = 4 ) ) %>%
    as_tibble() %>%
    t() %>%
    magrittr::set_colnames(c("xmin", "ymin", "xmax", "ymax")) %>%
    as_tibble() %>%
    select(xmin, xmax, ymin, ymax)
  
  cbind(frame, bndbx, classes) %>%
    as_tibble %>%
    mutate(frame = as.character(frame))
  }


classes <- c( "eyes", "nose", "mouth" )

if( ! file.exists( dataFile ) )
  {
  data <- list.files( annotationsDirectory, full.names = TRUE ) %>%
    discard( !str_detect( ., "xml" ) ) %>%
    map( ., read_xml ) %>%
    map_dfr( parseXML, classes )

  data <- list.files( annotationsDirectory, full.names = TRUE ) %>%
    discard( !str_detect( ., "xml" ) ) %>%
    map( ., read_xml ) %>%
    map_dfr( parseXML, classes )
  data <- data[complete.cases( data ),]

  write.csv( data, dataFile, row.names = FALSE )
  } else {
  data <- read.csv( dataFile )  
  }
uniqueImageFiles <- levels( as.factor( data$frame ) )

###
#
# Read in the testing data.  There are 1000 total images.  We used 800
# for training.  We now read the remaining data for testing/prediction.
#


count <- 1
for( i in ( numberOfTrainingData + 1 ):
  ( numberOfTrainingData + numberOfTestingData ) )
  {
  testingImageFiles[count] <- paste0( imageDirectory, uniqueImageFiles[i] )  
  count <- count + 1
  }

inputImageSize <- c( 250, 250 )
testingData <- array( dim = c( numberOfTestingData, inputImageSize, 3 ) )

cat( "Reading images...\n" )
pb <- txtProgressBar( min = 0, max = numberOfTestingData, style = 3 )
for ( i in 1:length( testingImageFiles ) )
  {
  # cat( "Reading ", testingImageFiles[i], "\n" )
  testingImage <- readJPEG( testingImageFiles[i] )

  r <- as.matrix( as.antsImage( testingImage[,,1] ) )
  g <- as.matrix( as.antsImage( testingImage[,,2] ) )
  b <- as.matrix( as.antsImage( testingImage[,,3] ) )
  
  r <- ( r - min( r ) ) / ( max( r ) - min( r ) )
  g <- ( g - min( g ) ) / ( max( g ) - min( g ) )
  b <- ( b - min( b ) ) / ( max( b ) - min( b ) )
  
  testingData[i,,,1] <- r
  testingData[i,,,2] <- g
  testingData[i,,,3] <- b

  if( i %% 100 == 0 )
    {
    gc( verbose = FALSE )
    }

  setTxtProgressBar( pb, i )
  }
cat( "\nDone.\n" )

X_test <- testingData

###
#
# Create the Y encoding for the test data
#

groundTruthLabels <- list()
for( i in 1:numberOfTestingData )
  {
  index <- numberOfTrainingData + i
  groundTruthBoxes <- data[which( data$frame == uniqueImageFiles[index] ),]

  image <- readJPEG( testingImageFiles[i] )
  groundTruthBoxes <- 
   data.frame( groundTruthBoxes[, 6], groundTruthBoxes[, 2:5] )
  colnames( groundTruthBoxes ) <- c( "class_id", 'xmin', 'xmax', 'ymin', 'ymax' )
  groundTruthLabels[[i]] <- groundTruthBoxes

  if( visuallyInspectEachImage == TRUE )
    {
    cat( "Drawing", testingImageFiles[i], "\n" )

    classIds <- groundTruthBoxes[, 1]

    boxColors <- c()
    boxCaptions <- c()
    for( j in 1:length( classIds ) )
      {
      boxColors[j] <- rainbow( 
        length( classes ) )[which( classes[classIds[j]] == classes )]
      boxCaptions[j] <- classes[which( classes[classIds[j]] == classes )]
      }
    image <- array( 0, dim = c( inputImageSize, 3 ) )
    for( k in 1:3 )
      {
      image[,,k] <- ( testingData[i,,,k] - min( testingData[i,,,k] ) ) / 
        ( max( testingData[i,,,k] ) - min( testingData[i,,,k] ) )
      }  
    drawRectangles( image, groundTruthBoxes[, 2:5], boxColors = boxColors, 
      captions = boxCaptions )
    readline( prompt = "Press [enter] to continue " )
    }
  }

if( visuallyInspectEachImage == TRUE )
  {
  cat( "\n\nDone inspecting images.\n" )
  }
 
###
#
# Create the SSD model
#

ssdOutput <- createSsd7Model2D( c( inputImageSize, 3 ), 
  numberOfClassificationLabels = length( classes ) + 1,
  aspectRatiosPerLayer = 
    list( c( 1.0, 2.0, 0.5 ),  
          c( 1.0, 2.0, 0.5 ),
          c( 1.0, 2.0, 0.5 ),
          c( 1.0, 2.0, 0.5 )
        )
  )

ssdModelTest <- ssdOutput$ssdModel 
anchorBoxes <- ssdOutput$anchorBoxes

Y_test <- encodeY( groundTruthLabels, anchorBoxes, inputImageSize, rep( 1.0, 4 ) )

###
#
#  Debugging:  visualize corresponding anchorBoxes
#

numberOfClassificationLabels <- length( classes ) + 1

if( visuallyInspectEachImage == TRUE )
  {
  for( i in 1:numberOfTestingData )
    {
    cat( "Drawing", testingImageFiles[i], "\n" )
    image <- array( 0, dim = c( inputImageSize, 3 ) )
    for( k in 1:3 )
      {
      image[,,k] <- ( testingData[i,,,k] - min( testingData[i,,,k] ) ) / 
        ( max( testingData[i,,,k] ) - min( testingData[i,,,k] ) )
      }  

    # Get anchor boxes  
    singleY <- Y_test[i,,]
    singleY <- singleY[which( rowSums( 
      singleY[, 2:( 1 + length( classes ) )] ) > 0 ),]

    xIndices <- numberOfClassificationLabels + 5:6
    singleY[, xIndices] <- singleY[, xIndices] * ( inputImageSize[1] - 2 ) + 1
    yIndices <- numberOfClassificationLabels + 7:8
    singleY[, yIndices] <- singleY[, yIndices] * ( inputImageSize[2] - 2 ) + 1

    anchorClassIds <- max.col( singleY[, 1:4] ) - 1

    anchorBoxColors <- c()
    anchorBoxCaptions <- c()
    for( j in 1:length( anchorClassIds ) )
      {
      anchorBoxColors[j] <- rainbow( 
        length( classes ) )[which( classes[anchorClassIds[j]] == classes )]
      # anchorBoxCaptions[j] <- classes[which( classes[anchorClassIds[j]] == classes )]
      }

    # Get truth boxes
    truthLabel <- groundTruthLabels[[i]]
    truthClassIds <- truthLabel[, 1]
    truthColors <- c()
    truthCaptions <- c()
    for( j in 1:length( truthClassIds ) )
      {
      truthColors[j] <- rainbow( 
        length( classes ) )[which( classes[truthClassIds[j]] == classes )]
      truthCaptions[j] <- classes[which( classes[truthClassIds[j]] == classes )]
      }

    boxes <- rbind( singleY[, 9:12], as.matrix( truthLabel[, 2:5] ) )
    boxColors <- c( anchorBoxColors, truthColors )
    confidenceValues <- c( rep( 0.2, length( anchorBoxColors ) ), rep( 1.0, length( truthColors ) ) )

    drawRectangles( image, boxes, boxColors = boxColors, confidenceValues = confidenceValues )

    readline( prompt = "Press [enter] to continue " )
    }
  }  


load_model_weights_hdf5( ssdModelTest, 
  filepath = paste0( baseDirectory, 'ssd7Weights.h5' ) )

# load_model_weights_hdf5( ssdModelTest, 
#   filepath = '/Users/ntustison/Pkg/ssdkeras/checkpoints3.h5' )

optimizerAdam <- optimizer_adam( 
  lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = 5e-04 )

ssdLoss <- lossSsd$new( backgroundRatio = 3L, minNumberOfBackgroundBoxes = 0L, 
  alpha = 1.0, numberOfClassificationLabels = length( classes ) + 1 )

ssdModelTest %>% compile( loss = ssdLoss$compute_loss, optimizer = optimizerAdam )

testingMetrics <- ssdModelTest %>% evaluate( X_test, Y_test )

X_test <- testingData

predictedData <- ssdModelTest %>% predict( X_test, verbose = 1 )
predictedDataDecoded <- decodeY( predictedData, inputImageSize, 
  confidenceThreshold = 0.1, overlapThreshold = NULL )

for( i in 1:length( predictedDataDecoded ) )
  {
  cat( "Drawing", testingImageFiles[i], "\n" )
  image <- array( 0, dim = c( inputImageSize, 3 ) )
  for( k in 1:3 )
    {
    image[,,k] <- ( testingData[i,,,k] - min( testingData[i,,,k] ) ) / 
      ( max( testingData[i,,,k] ) - min( testingData[i,,,k] ) )
    }  

  boxes <- predictedDataDecoded[[i]][, 3:6]
  classIds <- predictedDataDecoded[[i]][, 1]
  confidenceValues <- predictedDataDecoded[[i]][, 2]

  boxColors <- c()
  boxCaptions <- c()
  for( j in 1:length( classIds ) )
    {
    boxColors[j] <- rainbow( 
      length( classes ) )[which( classes[classIds[j]] == classes )]
    boxCaptions[j] <- classes[which( classes[classIds[j]] == classes )]
    }
  drawRectangles( image, boxes, boxColors = boxColors, captions = boxCaptions,
    confidenceValues = confidenceValues )

  readline( prompt = "Press [enter] to continue\n" )
  }


# image <- readJPEG( testingImageFiles[1] )
# for( i in 1:dim( predictedData )[2] )
#   {
#   cat( "Drawing box", i, "\n" )

#   boxes <- matrix( predictedData[1,, 5:8], ncol = 4 )
#   drawRectangles( image, boxes, boxColors = "red" )
#   cat( "   back : ", predictedData[1, i, 1], "\n" )
#   for( j in 1:length( classes ) )
#     {
#     cat( "  ", classes[j], ": ", predictedData[1, i, j+1], "\n" )
#     }
#   readline( prompt = "Press [enter] to continue\n" )
#   }
