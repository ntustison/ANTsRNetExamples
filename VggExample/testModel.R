library( ANTsR )
library( keras )
library( abind )
library( ggplot2 )

# Dog vs. cat data available from here:
#    https://www.kaggle.com/c/dogs-vs-cats/data
# Also use the human faces from:
#    http://www.vision.caltech.edu/Image_Datasets/Caltech_10K_WebFaces/

testingProportion <- 0.01
testingImageSize <- c( 100, 100 )

baseDirectory <- './'
dataDirectory <- paste0( baseDirectory, 'Images/' )
modelDirectory <- paste0( baseDirectory, '../Models/' )

source( paste0( modelDirectory, 'createVggModel.R' ) )

# Yeah, I know I'm double-dipping here but I'm just trying to get something
# to work at this point.
testingDirectories <- c()
testingDirectories[1] <- paste0( dataDirectory, 'TrainingDataDog/' )
testingDirectories[2] <- paste0( dataDirectory, 'TrainingDataHuman/' )

numberOfSubjectsPerCategory <- 1e6
for( i in 1:length( testingDirectories ) )
  {
  testingImageFilesPerCategory <- list.files( 
    path = testingDirectories[i], pattern = "*.jpg", full.names = TRUE )
  numberOfSubjectsPerCategory <- min( numberOfSubjectsPerCategory,
    testingProportion * length( testingImageFilesPerCategory ) )
  }

testingImageFiles <- c()
testingClassifications <- c()
for( i in 1:length( testingDirectories ) )
  {
  testingImageFilesPerCategory <- list.files( 
    path = testingDirectories[i], pattern = "*.jpg", full.names = TRUE )

  set.seed( 567 )
  testingIndices <- sample.int( 
    length( testingImageFilesPerCategory ), size = numberOfSubjectsPerCategory )
  testingImageFiles <- append( 
    testingImageFiles, testingImageFilesPerCategory[testingIndices] )  
  testingClassifications <- append( testingClassifications, 
    rep( i-1, length( testingIndices ) ) )
  }
  

testingImages <- list()
testingImageArrays <- list()
for ( i in 1:length( testingImageFiles ) )
  {
  cat( "Reading ", testingImageFiles[i], "(", i, " of ", length( testingImageFiles ), ")\n" )
  testingImages[[i]] <- resampleImage( 
    antsImageRead( testingImageFiles[i], dimension = 2 ),
    testingImageSize, useVoxels = TRUE )
  testingImageArrays[[i]] <- as.array( testingImages[[i]] )
  }

testingData <- abind( testingImageArrays, along = 3 )  
testingData <- aperm( testingData, c( 3, 1, 2 ) )
testingData <- ( testingData - mean( testingData ) ) / sd( testingData )

X_test <- array( testingData, dim = c( dim( testingData ), 1 ) )
segmentationLabels <- sort( unique( testingClassifications ) )
numberOfLabels <- length( segmentationLabels )
Y_test <- to_categorical( testingClassifications, numberOfLabels )

numberOfLabels <- length( unique( as.vector( testingClassifications ) ) )

vggModelTest <- createVggModel2D( c( dim( testingImageArrays[[1]] ), 1 ), layers = c( 1, 2, 3, 4 ),
  numberOfClassificationLabels = numberOfLabels, style = 16 )
load_model_weights_hdf5( vggModelTest, filepath = paste0( baseDirectory, 'vggWeights.h5' ) )

testingMetrics <- vggModelTest %>% evaluate( X_test, Y_test )
predictedData <- vggModelTest %>% predict( X_test, verbose = 1 )

