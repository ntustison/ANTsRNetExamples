library( ANTsR )
library( xml2 )
library( tidyverse )
library( stringr )
library( keras )
library( ggplot2 )
library( jpeg )

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

baseDirectory <- './'
dataDirectory <- paste0( baseDirectory, './lfw_faces_tagged/' )
imageDirectory <- paste0( dataDirectory, 'Images/' )
annotationsDirectory <- paste0( dataDirectory, 'Annotations/' )
dataFile <- paste0( dataDirectory, 'data.csv' )

modelDirectory <- paste0( baseDirectory, '../../Models/' )

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

###
#
# Read in the testing data.  There are 1000 total images.  We used 800
# for training.  We now read the remaining data for testing/prediction.
#

numberOfTrainingData <- 800
numberOfTestingData <- 200
testingImageFiles <- rep( NA, numberOfTestingData )

count <- 1
for( i in ( numberOfTrainingData + 1 ):
  ( numberOfTrainingData + numberOfTestingData ) )
  {
  testingImageFiles[count] <- paste0( imageDirectory, data$frame[i] )  
  count <- count + 1
  }

inputImageSize <- c( 300, 300 )
testingData <- array( dim = c( numberOfTestingData, inputImageSize, 3 ) )

cat( "Reading images...\n" )
pb <- txtProgressBar( min = 0, max = numberOfTestingData, style = 3 )
for ( i in 1:length( testingImageFiles ) )
  {
  # cat( "Reading ", testingImageFiles[i], "\n" )
  testingImage <- readJPEG( testingImageFiles[i] )

  r <- as.matrix( resampleImage( 
        as.antsImage( testingImage[,,1] ), 
        inputImageSize, useVoxels = TRUE ) )
  r <- ( r - mean( r ) ) / sd( r )          
  g <- as.matrix( resampleImage( 
        as.antsImage( testingImage[,,2] ), 
        inputImageSize, useVoxels = TRUE ) )
  g <- ( g - mean( g ) ) / sd( g )      
  b <- as.matrix( resampleImage( 
        as.antsImage( testingImage[,,3] ), 
        inputImageSize, useVoxels = TRUE ) )
  b <- ( b - mean( b ) ) / sd( b )      

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
uniqueImageFiles <- levels( as.factor( data$frame ) )

groundTruthLabels <- list()
for( i in 1:numberOfTestingData )
  {
  index <- numberOfTrainingData + i
  groundTruthBoxes <- data[which( data$frame == uniqueImageFiles[index] ),]
  groundTruthBoxes <- 
    data.frame( groundTruthBoxes[, 6], groundTruthBoxes[, 2:5] )
  colnames( groundTruthBoxes ) <- c( "class_id", 'xmin', 'xmax', 'ymin', 'ymax' )
  groundTruthLabels[[i]] <- groundTruthBoxes
  }

Y_test <- encodeY( groundTruthLabels, anchorBoxes, rep( 1.0, 4 ) )


###
#
# Create the SSD model
#

source( paste0( modelDirectory, 'createSsdModel.R' ) )

# Input size must be greater than >= 258 for a single dimension

inputImageSize <- c( inputImageSize, 3 )
ssdOutput <- createSsdModel2D( inputImageSize, 
  numberOfClassificationLabels = length( classes ) + 1,
  )

ssdModelTest <- ssdOutput$ssdModel 
anchorBoxes <- ssdOutput$anchorBoxes

load_model_weights_hdf5( ssdModelTest, 
  filepath = paste0( baseDirectory, 'ssdWeights.h5' ) )

optimizerAdam <- optimizer_adam( 
  lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = 5e-05 )

ssdLoss <- lossSsd$new( backgroundRatio = 3L, minNumberOfBackgroundBoxes = 0L, 
  alpha = 1.0, numberOfClassificationLabels = length( classes ) + 1 )

ssdModelTest %>% compile( loss = ssdLoss$compute_loss, optimizer = optimizerAdam )

testingMetrics <- ssdModelTest %>% evaluate( X_test, Y_test )

predictedData <- ssdModelTest %>% predict( X_test, verbose = 1 )
predictedDataDecoded <- decodeY( predictedData )
