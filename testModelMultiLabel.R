library( ANTsR )
library( keras )
library( abind )
library( ggplot2 )

baseDirectory <- './'
dataDirectory <- paste0( baseDirectory, 'Images/' )
testingDirectory <- paste0( dataDirectory, 'TestingData/' )
predictedDirectory <- paste0( dataDirectory, 'PredictedData/' )
dir.create( predictedDirectory )


testingImageFiles <- list.files( path = testingDirectory, pattern = "H1_2D", full.names = TRUE )
testingMaskFiles <- list.files( path = testingDirectory, pattern = "Mask_2D", full.names = TRUE )

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
  testingMaskArrays[[i]][which( testingMaskArrays[[i]] > 1 )] <- 1
  }

testingData <- abind( testingImageArrays, along = 3 )  
testingData <- aperm( testingData, c( 3, 1, 2 ) )
testingData <- ( testingData - mean( testingData ) ) / sd( testingData )

X_test <- array( testingData, dim = c( dim( testingData ), 1 ) )

testingLabelData <- abind( testingMaskArrays, along = 3 )  
testingLabelData <- aperm( testingLabelData, c( 3, 1, 2 ) )

numberOfLabels <- length( unique( as.vector( testingLabelData ) ) )

# Different implementation of keras::to_categorical().  The ordering 
# of the array elements seems to get screwed up.

Y_test <- testingLabelData
Y_test[which( testingLabelData == 0)] <- 1
Y_test[which( testingLabelData != 0)] <- 0

for( i in 2:numberOfLabels )
  {
  Y_test_label <- testingLabelData 
  Y_test_label[which( testingLabelData == segmentationLabels[i] )] <- 1
  Y_test_label[which( testingLabelData != segmentationLabels[i] )] <- 0

  Y_test <- abind( Y_test, Y_test_label, along = 4 )
  }

unetModelTest <- createUnetModel2D( c( dim( testingImageArrays[[1]] ), 1 ), numberOfClassificationLabels = numberOfLabels, layers = 1:5 )
load_model_weights_hdf5( unetModelTest, filepath = paste0( baseDirectory, 'unetModelMultiLabelWeights.h5' ) )

testingMetrics <- unetModelTest %>% evaluate( X_test, Y_test )

predictedData <- unetModelTest %>% predict( X_test, verbose = 1 )

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

for( i in 1:1 )
  {
  imageArray <- X_train[i,,,1]  
  image <- as.antsImage( imageArray, reference = trainingImages[[i]] )

  imageFileName <- gsub( ".nii.gz", paste0( "_Recreated", j, ".nii.gz" ), trainingImageFiles[[i]] )
  imageFileName <- gsub( trainingDirectory, predictedDirectory, imageFileName )

  antsImageWrite( image, imageFileName )

  for( j in 1:numberOfLabels )
    {
    imageArray <- Y_train[i,,,j]  
    image <- as.antsImage( imageArray, reference = trainingImages[[i]] )

    imageFileName <- gsub( ".nii.gz", paste0( "_Probability", j, ".nii.gz" ), trainingImageFiles[[i]] )
    imageFileName <- gsub( trainingDirectory, predictedDirectory, imageFileName )

    antsImageWrite( image, imageFileName ) 
    }  
  }

