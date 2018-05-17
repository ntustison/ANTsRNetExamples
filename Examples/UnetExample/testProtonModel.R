library( ANTsR )
library( ANTsRNet )
library( keras )
library( abind )
library( ggplot2 )

keras::backend()$clear_session()

baseDirectory <- './'
dataDirectory <- paste0( baseDirectory, 'Images/' )
testingDirectory <- paste0( dataDirectory, 'TestingData/' )
predictedDirectory <- paste0( dataDirectory, 'PredictedData/' )
dir.create( predictedDirectory )

testingImageFiles <- list.files( 
  path = testingDirectory, pattern = "N4Denoised_2D", full.names = TRUE )
testingMaskFiles <- list.files( 
  path = testingDirectory, pattern = "Mask_2D", full.names = TRUE )

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
  # testingMaskArrays[[i]][which( testingMaskArrays[[i]] > 1 )] <- 1
  }

testingData <- abind( testingImageArrays, along = 3 )  
testingData <- aperm( testingData, c( 3, 1, 2 ) )
testingData <- ( testingData - mean( testingData ) ) / sd( testingData )

X_test <- array( testingData, dim = c( dim( testingData ), 1 ) )

testingLabelData <- abind( testingMaskArrays, along = 3 )  
testingLabelData <- aperm( testingLabelData, c( 3, 1, 2 ) )

segmentationLabels <- sort( unique( as.vector( testingLabelData ) ) )
numberOfLabels <- length( unique( as.vector( testingLabelData ) ) )

Y_test <- encodeUnet( testingLabelData, segmentationLabels )

unetModelTest <- createUnetModel2D( c( dim( testingImageArrays[[1]] ), 1 ), 
  numberOfClassificationLabels = 3, convolutionKernelSize = c( 5, 5 ),
  deconvolutionKernelSize = c( 5, 5 ), lowestResolution = 32, dropoutRate = 0.2 )
load_model_weights_hdf5( unetModelTest, 
  filepath = paste0( baseDirectory, 'unetWeights.h5' ) )
unetModelTest %>% compile( loss = loss_multilabel_dice_coefficient_error,
  optimizer = optimizer_adam( lr = 0.0001 ),  
  metrics = c( multilabel_dice_coefficient ) )

testingMetrics <- unetModelTest %>% evaluate( X_test, Y_test )

predictedData <- unetModelTest %>% predict( X_test, verbose = 1 )

probabilityImages <- decodeUnet( predictedData, testingImages[[1]] )

for( i in 1:length( probabilityImages ) )
  {
  for( j in 1:length( probabilityImages[[i]] ) )
    {
    imageFileName <- gsub( ".nii.gz", 
      paste0( "_Probability", segmentationLabels[j], ".nii.gz" ), 
      testingImageFiles[[i]] )
    imageFileName <- 
      gsub( testingDirectory, predictedDirectory, imageFileName )

    probabilityArray <- as.array( probabilityImages[[i]][[j]] )

    antsImageWrite( 
      as.antsImage( probabilityArray, reference = testingImages[[i]] ), 
      imageFileName ) 
    }  
  }
