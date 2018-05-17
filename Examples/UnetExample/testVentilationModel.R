library( ANTsR )
library( ANTsRNet )
library( keras )

keras::backend()$clear_session()

baseDirectory <- './'
dataDirectory <- paste0( baseDirectory, 'Images/Ventilation/' )
testingDirectory <- paste0( dataDirectory, 'TestingData/' )
predictedDirectory <- paste0( dataDirectory, 'PredictedData/' )
dir.create( predictedDirectory )

source( paste0( baseDirectory, 'unetVentilationBatchGenerator.R' ) )

classes <- c( "background", "ventilation defect", "hypo-ventilation", "normal ventilation", "hyper-normal ventilation" )
numberOfClassificationLabels <- length( classes )

imageMods <- c( "Ventilation", "Foreground mask" )
channelSize <- length( imageMods )

resampledImageSize <- c( 80, 128 )

unetModel <- createUnetModel2D( c( resampledImageSize, channelSize ), 
  numberOfClassificationLabels = numberOfClassificationLabels, 
  convolutionKernelSize = c( 5, 5 ),
  deconvolutionKernelSize = c( 5, 5 ), lowestResolution = 32,
  dropoutRate = 0.2 )
load_model_weights_hdf5( unetModel, 
  filepath = paste0( dataDirectory, 'unetWeights.h5' ) )
unetModel %>% compile( loss = loss_multilabel_dice_coefficient_error,
  optimizer = optimizer_adam( lr = 0.0001 ),  
  metrics = c( multilabel_dice_coefficient ) )

testingImageDirectory <- paste0( dataDirectory, 'TestingData/' )
testingImageFiles <- list.files( 
  path = testingImageDirectory, pattern = "N4_Denoised", full.names = TRUE )
testingMaskFiles <- list.files( 
  path = testingImageDirectory, pattern = "Mask", full.names = TRUE )

for( i in 1:length( testingImageFiles ) )
  {
  subjectId <- basename( testingImageFiles[i] )
  subjectId <- sub( "N4_Denoised.nii.gz", '', subjectId )

  image <- antsImageRead( testingImageFiles[i], dimension = 2 )
  imageSize <- dim( image )
  resampledImage <- resampleImage( image, resampledImageSize, 
    useVoxels = TRUE, interpType = 1 )
  resampledImageArray <- as.array( resampledImage )  
  resampledImageArray <- ( resampledImageArray - mean( resampledImageArray ) ) / 
    sd( resampledImageArray )

  mask <- antsImageRead( testingMaskFiles[i], dimension = 2 )
  resampledMask <- resampleImage( mask, resampledImageSize, 
    useVoxels = TRUE, interpType = 1 )
  resampledMaskArray <- as.array( resampledMask )  

  batchX <- array( data = 0, 
    dim = c( 1, resampledImageSize, channelSize ) )

  batchX[1,,,1] <- resampledImageArray
  batchX[1,,,2] <- resampledMaskArray

  predictedData <- unetModel %>% predict( batchX, verbose = 0 )
  probabilityImagesArray <- decodeUnet( predictedData, image )

  for( j in seq_len( numberOfClassificationLabels ) )
    {
    imageFileName <- paste0( 
      predictedDirectory, subjectId, "Probability", j, ".nii.gz" )

    cat( "Writing", imageFileName, "\n" )  

    probabilityArray <- as.array( 
      resampleImage( probabilityImagesArray[[1]][[j]], 
        imageSize, useVoxels = TRUE, interpType = 1 ) )
    
    antsImageWrite( as.antsImage( probabilityArray, reference = image ),
      imageFileName )  
    }  
  }
