library( ANTsR )
library( keras )
library( abind )
library( ggplot2 )

baseDirectory <- '/Users/ntustison/Data/UNet/'
dataDirectory <- paste0( baseDirectory, 'Images/' )
testingDirectory <- paste0( dataDirectory, 'TestingData/' )
predictedDirectory <- paste0( dataDirectory, 'PredictedData/' )

testingImageFiles <- list.files( path = testingDirectory, pattern = "H1_2D", full.names = TRUE )
testingMaskFiles <- list.files( path = testingDirectory, pattern = "BinaryMask_2D", full.names = TRUE )

testingImages <- list()
testingMasks <- list()
testingImageArrays <- list()
testingMaskArrays <- list()

for ( i in 1:length( testingImageFiles ) )
  {
  testingImages[[i]] <- antsImageRead( testingImageFiles[i], dimension = 2 )    
  testingMasks[[i]] <- antsImageRead( testingMaskFiles[i], dimension = 2 )    

  testingImageArrays[[i]] <- as.array( testingImages[[i]] )
  testingMaskArrays[[i]] <- as.array( testingMasks[[i]] )  
  }

testingData <- abind( testingImageArrays, along = 3 )  
testingData <- aperm( testingData, c( 3, 1, 2 ) )

testingLabelData <- abind( testingMaskArrays, along = 3 )  
testingLabelData <- aperm( testingLabelData, c( 3, 1, 2 ) )

numberOfLabels <- 2

X_test <- array( testingData, dim = c( dim( testingData ), numberOfLabels ) )
Y_test <- array( to_categorical( testingLabelData ), dim = c( dim( testingData ), numberOfLabels ) )

unetModelTest <- createUnetModel2D( dim( testingImageArrays[[1]] ), numberOfClassificationLabels = numberOfLabels, layers = 1:3 )
load_model_weights_hdf5( unetModelTest, filepath = paste0( baseDirectory, 'unetModelWeights.h5' ) )

testingMetrics <- unetModelTest %>% evaluate( X_test, Y_test )

predictedData <- unetModelTest %>% predict( X_test )

numberOfTestingImages <- dim( predictedData )[1]

for( i in 1:numberOfTestingImages )
  {
  for( j in 1:numberOfLabels )
    {
    imageArray <- predictedData[i,,,j]  
    image <- as.antsImage( imageArray, reference = testingImages[[i]] )

    imageFileName <- gsub( ".nii.gz", paste0( "_Probability", j, ".nii.gz" ), testingImageFiles[[i]] )
    imageFileName <- gsub( testingDirectory, predictedDirectory, imageFileName )

    antsImageWrite( image, imageFileName ) 
    }  
  }


