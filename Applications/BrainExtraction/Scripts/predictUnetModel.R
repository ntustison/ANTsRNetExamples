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
  "S_template3_resampled.nii.gz" ), dimension = 2 )
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
  image <- antsImageRead( brainImageFiles[i], dimension = 3 )

  subjectId <- basename( brainImageFiles[i] )
  subjectId <- sub( ".nii.gz", '', subjectId )

  antsXfrms <- antsRegistration( fixed = reorientTemplate, moving = image, 
    typeofTransform = "QuickRigid" )

  warpedImage <- antsApplyTransforms( fixed = reorientTemplate, 
    moving = image, transformlist = antsXfrms$fwdtransforms )

  batchX <- array( data = as.array( warpedImage , 
    dim = c( 1, resampledImageSize, channelSize ) )
    
  predictedData <- unetModel %>% predict( batchX, verbose = 0 )
  probabilityImagesArray <- decodeUnet( predictedData, warpedImage )

  for( j in seq_len( numberOfClassificationLabels ) )
    {
    imageFileName <- paste0( 
      evaluationDirectory, subjectId, "Probability", j, ".nii.gz" )

    cat( "Writing", imageFileName, "\n" )  

    probabilityImage <- antsApplyTransforms( fixed = image,
      moving = probabilityImagesArray[[1]][[j]], 
      transformlist = antsXfrms$invtransforms )

    antsImageWrite( probabilityImage, imageFileName )  
    }  
  }






