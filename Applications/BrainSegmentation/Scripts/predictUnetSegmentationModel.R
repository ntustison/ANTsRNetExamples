library( ANTsR )
library( ANTsRNet )
library( keras )

keras::backend()$clear_session()

baseDirectory <- '../'
dataDirectory <- paste0( baseDirectory, '../BrainExtraction/Data/' )
evaluationDirectory <- paste0( baseDirectory, 'Data/Results/' )

classes <- c( "background", "CSF", "GM", "WM", "DGM", "BrainStem", "Cerebellum" )

numberOfClassificationLabels <- length( classes )

imageMods <- c( "T1", "ExtractionMask" )
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
brainMaskFiles <- list.files( path = paste0( dataDirectory, "Results/" ),
  pattern = "*.nii.gz", full.names = TRUE )

for( i in 1:length( brainImageFiles ) )
  {
  startTimeTotal <- Sys.time()  

  cat( "Reading", brainImageFiles[i] )  
  startTime <- Sys.time()
  image <- antsImageRead( brainImageFiles[i], dimension = 3 )
  mask <- antsImageRead( brainMaskFiles[i], dimension = 3 )
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
  warpedMask <- applyAntsrTransformToImage( xfrm, mask, reorientTemplate )
  endTime <- Sys.time()  
  elapsedTime <- endTime - startTime
  cat( " (elapsed time:", elapsedTime, "seconds)\n" )

  warpedArray <- as.array( warpedImage )
  warpedArray <- ( warpedArray - mean( warpedArray ) ) / sd( warpedArray )

  batchX <- array( data = 0, dim = c( 1, resampledImageSize, channelSize ) )
  batchX[1,,,,1] <- warpedArray
  batchX[1,,,,2] <- as.array( warpedMask )

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
  endTime <- Sys.time()  
  elapsedTime <- endTime - startTime
  cat( " (elapsed time:", elapsedTime, "seconds)\n" )

  cat( "Writing probability images...\n" )  
  for( j in seq_len( numberOfClassificationLabels ) )
    {
    imageFileName <- paste0( 
      evaluationDirectory, subjectId, "Probability", j-1, ".nii.gz" )

    probabilityImage <- applyAntsrTransformToImage( invertAntsrTransform( xfrm ),
      probabilityImagesArray[[1]][[j]], image )

    antsImageWrite( probabilityImage, imageFileName )  
    }  

  endTimeTotal <- Sys.time()  
  elapsedTimeTotal <- endTimeTotal - startTimeTotal
  cat( "\nTotal elapsed time:", elapsedTimeTotal, "seconds\n\n" )
  }
