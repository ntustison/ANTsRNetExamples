library( ANTsR )
library( ANTsRNet )
library( keras )
library( jpeg )

visuallyInspectEachImage <- FALSE

numberOfTrainingData <- 900
numberOfTestingData <- 100
testingImageFiles <- rep( NA, numberOfTestingData )

baseDirectory <- './'
dataDirectory <- paste0( baseDirectory, './lfw_faces_tagged/' )
imageDirectory <- paste0( dataDirectory, 'Images/' )
annotationsDirectory <- paste0( dataDirectory, 'Annotations/' )
dataFile <- paste0( dataDirectory, 'data.csv' )

classes <- c( "eyes", "nose", "mouth" )

data <- read.csv( dataFile )  
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
  numberOfClassificationLabels = length( classes ) + 1
  )

ssdModelTest <- ssdOutput$ssdModel 
anchorBoxes <- ssdOutput$anchorBoxes

Y_test <- encodeSsd2D( groundTruthLabels, anchorBoxes, inputImageSize, rep( 1.0, 4 ) )

load_model_weights_hdf5( ssdModelTest, 
  filepath = paste0( baseDirectory, 'ssd7Weights.h5' ) )

optimizerAdam <- optimizer_adam( 
  lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = 5e-04 )

ssdLoss <- LossSSD$new( backgroundRatio = 3L, minNumberOfBackgroundBoxes = 0L, 
  alpha = 1.0, numberOfClassificationLabels = length( classes ) + 1 )

ssdModelTest %>% compile( loss = ssdLoss$compute_loss, optimizer = optimizerAdam )

X_test <- testingData
testingMetrics <- ssdModelTest %>% evaluate( X_test, Y_test )
predictedData <- ssdModelTest %>% predict( X_test, verbose = 1 )
predictedDataDecoded <- decodeSsd2D( predictedData, inputImageSize, 
  confidenceThreshold = 0.4, overlapThreshold = 0.4 )

for( i in 1:length( predictedDataDecoded ) )
  {
  cat( "Drawing", testingImageFiles[i], "\n" )
  image <- array( 0, dim = c( inputImageSize, 3 ) )
  for( k in 1:3 )
    {
    image[,,k] <- ( testingData[i,,,k] - min( testingData[i,,,k] ) ) / 
      ( max( testingData[i,,,k] ) - min( testingData[i,,,k] ) )
    }  

  # predictedDataDecoded contains all the boxes exceeding some 
  # sorting constraing.  Here we choose the best box for each 
  # class.

  maxIndices <- c()
  classIds <- unique( predictedDataDecoded[[i]][, 1] )
  for( j in 1:length( classIds ) )
    {
    classIndices <- which( predictedDataDecoded[[i]][ , 1] == classIds[j] )
    maxIndex <- which.max( predictedDataDecoded[[i]][classIndices, 2] )
    maxIndices[j] <- classIndices[maxIndex]
    }
  boxes <- predictedDataDecoded[[i]][maxIndices, 3:6]
  confidenceValues <- predictedDataDecoded[[i]][maxIndices, 2]

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

