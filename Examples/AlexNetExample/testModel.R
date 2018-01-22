library( ANTsR )
library( keras )
library( abind )
library( ggplot2 )

# Dog vs. cat data available from here:
#    https://www.kaggle.com/c/dogs-vs-cats/data
# Also use the human faces from:
#    http://www.vision.caltech.edu/Image_Datasets/Caltech_10K_WebFaces/

testingProportion <- 0.01
testingImageSize <- c( 227, 227 )

baseDirectory <- './'
dataDirectory <- paste0( baseDirectory, './Images/' )
modelDirectory <- paste0( baseDirectory, '../../Models/' )

source( paste0( modelDirectory, 'createAlexNetModel.R' ) )

# Yeah, I know I'm double-dipping here but I'm just trying to get something
# to work at this point.
testingDirectories <- c()
testingDirectories[1] <- paste0( dataDirectory, 'TrainingDataPlanes/' )
testingDirectories[2] <- paste0( dataDirectory, 'TrainingDataHuman/' )
testingDirectories[3] <- paste0( dataDirectory, 'TrainingDataCat/' )
testingDirectories[4] <- paste0( dataDirectory, 'TrainingDataDog/' )

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

  set.seed( 1234 )
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
  cat( "Reading ", testingImageFiles[i], "\n" )
  testingImages[[i]] <- readJPEG( testingImageFiles[i] )
  if( length( dim( testingImages[[i]] ) ) == 3 )
    {
    r <- as.matrix( resampleImage( 
          as.antsImage( testingImages[[i]][,,1] ), 
          testingImageSize, useVoxels = TRUE ) )
    r <- ( r - mean( r ) ) / sd( r )          
    g <- as.matrix( resampleImage( 
          as.antsImage( testingImages[[i]][,,2] ), 
          testingImageSize, useVoxels = TRUE ) )
    g <- ( g - mean( g ) ) / sd( g )      
    b <- as.matrix( resampleImage( 
          as.antsImage( testingImages[[i]][,,3] ), 
          testingImageSize, useVoxels = TRUE ) )
    b <- ( b - mean( b ) ) / sd( b )      
    } else {
    r <- as.matrix( resampleImage( 
          as.antsImage( testingImages[[i]] ), 
          testingImageSize, useVoxels = TRUE ) )
    r <- ( r - mean( r ) ) / sd( r )      
    g <- b <- r  
    }      
  testingImageArrays[[i]] <- abind( r, g, b, along = 3 )  
  }

# testingData <- abind( testingImageArrays, along = 3 )  
# testingData <- aperm( testingData, c( 3, 1, 2 ) )
# X_test <- array( testingData, dim = c( dim( testingData ), 1 ) )

testingData <- abind( testingImageArrays, along = 4 )  
testingData <- aperm( testingData, c( 4, 1, 2, 3 ) )
X_test <- array( testingData, dim = c( dim( testingData ) ) )

segmentationLabels <- sort( unique( testingClassifications ) )
numberOfLabels <- length( segmentationLabels )
Y_test <- to_categorical( testingClassifications, numberOfLabels )

numberOfLabels <- length( unique( as.vector( testingClassifications ) ) )

# alexNetModelTest <- createAlexNetModel2D( c( dim( testingImageArrays[[1]] ), 1 ),
#   numberOfClassificationLabels = numberOfLabels, style = 19 )
alexNetModelTest <- createAlexNetModel2D( dim( testingImageArrays[[1]] ),
  numberOfClassificationLabels = numberOfLabels )
if( numberOfLabels == 2 )   
  {
  alexNetModelTest %>% compile( loss = 'binary_crossentropy',
    optimizer = optimizer_adam( lr = 0.0001 ),  
    metrics = c( 'binary_crossentropy', 'accuracy' ) )
  } else {
  alexNetModelTest %>% compile( loss = 'categorical_crossentropy',
    optimizer = optimizer_adam( lr = 0.0001 ),  
    metrics = c( 'categorical_crossentropy', 'accuracy' ) )
  }

load_model_weights_hdf5( alexNetModelTest, filepath = paste0( baseDirectory, 'alexNetWeights.h5' ) )

testingMetrics <- alexNetModelTest %>% evaluate( X_test, Y_test )
predictedData <- alexNetModelTest %>% predict( X_test, verbose = 1 )

