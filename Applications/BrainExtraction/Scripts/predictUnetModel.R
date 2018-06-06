library( ANTsR )
library( ANTsRNet )
library( keras )

keras::backend()$clear_session()

baseDirectory <- '../'
dataDirectory <- '../Data/'
evaluationDirectory <- '../Data/Results/'

classes <- c( "background", "brain" )
numberOfClassificationLabels <- length( classes )

imageMods <- c( "T1" )
channelSize <- length( imageMods )

reorientTemplate <- antsImageRead( paste0( dataDirectory, 
  "Template/S_template3_resampled.nii.gz" ), dimension = 3 )
resampledImageSize <- dim( reorientTemplate )

unetModel <- createUnetModel3D( c( resampledImageSize, channelSize ), 
  numberOfClassificationLabels = numberOfClassificationLabels, 
  numberOfLayers = 4, numberOfFiltersAtBaseLayer = 16, dropoutRate = 0.0,
  convolutionKernelSize = c( 5, 5, 5 ), 
  deconvolutionKernelSize = c( 5, 5, 5 ) )
load_model_weights_hdf5( unetModel, 
  filepath = paste0( baseDirectory, 'unetModelWeights.h5' ) )
unetModel %>% compile( loss = loss_multilabel_dice_coefficient_error,
  optimizer = optimizer_adam( lr = 0.0001 ),  
  metrics = c( multilabel_dice_coefficient ) )

brainImageFiles <- list.files( path = paste0( dataDirectory, "Images/" ),
  pattern = "*.nii.gz", full.names = TRUE )

for( i in 1:length( brainImageFiles ) )
  {
  startTimeTotal <- Sys.time()  

  cat( "Reading", brainImageFiles[i] )  
  startTime <- Sys.time()
  image <- antsImageRead( brainImageFiles[i], dimension = 3 )
  endTime <- Sys.time()  
  elapsedTime <- endTime - startTime
  cat( " (elapsed time:", elapsedTime, "seconds)\n" )

  subjectId <- basename( brainImageFiles[i] )
  subjectId <- sub( ".nii.gz", '', subjectId )

  cat( "Normalizing to template" )  
  startTime <- Sys.time()
  centerOfMassTemplate <- getCenterOfMass( reorientTemplate )
  centerOfMassImage <- getCenterOfMass( image )
  xfrm <- createAntsrTransform( type = "Euler3DTransform", 
    center = centerOfMassTemplate, 
    translation = centerOfMassImage - centerOfMassTemplate )
  warpedImage <- applyAntsrTransformToImage( xfrm, image, reorientTemplate )
  endTime <- Sys.time()  
  elapsedTime <- endTime - startTime
  cat( " (elapsed time:", elapsedTime, "seconds)\n" )

  batchX <- array( data = as.array( warpedImage ), 
    dim = c( 1, resampledImageSize, channelSize ) )

  batchX <- ( batchX - mean( batchX ) ) / sd( batchX )  
    
  cat( "Prediction and decoding" )  
  startTime <- Sys.time()
  predictedData <- unetModel %>% predict( batchX, verbose = 0 )
  probabilityImagesArray <- decodeUnet( predictedData, reorientTemplate )
  endTime <- Sys.time()  
  elapsedTime <- endTime - startTime
  cat( " (elapsed time:", elapsedTime, "seconds)\n" )

  imageFileName <- paste0( 
    evaluationDirectory, subjectId, "BrainMaskProbability.nii.gz" )

  cat( "Renormalize to native space" )  
  startTime <- Sys.time()
  probabilityImage <- applyAntsrTransformToImage( invertAntsrTransform( xfrm ),
    probabilityImagesArray[[1]][[2]], image )
  endTime <- Sys.time()  
  elapsedTime <- endTime - startTime
  cat( " (elapsed time:", elapsedTime, "seconds)\n" )

  cat( "Writing", imageFileName )
  startTime <- Sys.time()
  antsImageWrite( probabilityImage, imageFileName )  
  endTime <- Sys.time()  
  elapsedTime <- endTime - startTime
  cat( " (elapsed time:", elapsedTime, "seconds)\n" )

  endTimeTotal <- Sys.time()  
  elapsedTimeTotal <- endTimeTotal - startTimeTotal
  cat( "\nTotal elapsed time:", elapsedTimeTotal, "seconds\n\n" )
  }






