library( ANTsR )
library( ANTsRNet )
library( keras )

baseDir <- './'
inputImageDir <- paste0( baseDir, 'TestingImages/' )

outputDir <- paste0( baseDir, 'TestingResults/' )
dir.create( outputDir, showWarnings = FALSE )

inputImages <- list.files( path = inputImageDir, pattern = ".bmp",
  full.names = TRUE, recursive = TRUE )
numberOfInputImages <- length( inputImages )

highResolutionPatchSize <- c( 64, 64 )
lowResolutionPatchSize <- c( 32, 32 )
channelSize <- 1

###
#
# Create the super resolution model
#
srModel <- createImageSuperResolutionModel2D( 
 c( highResolutionPatchSize, channelSize ),
 convolutionKernelSizes = list( c( 9, 9 ), c( 1, 1 ), c( 5, 5 ) ) )
srModel %>% compile( loss = loss_mean_squared_error,
  optimizer = optimizer_adam( lr = 0.001 ),  
  metrics = c( 'mse' ) )
load_model_weights_hdf5( srModel, paste0( baseDir, "srWeights.h5" ) )

for( i in seq_len( numberOfInputImages ) )
  {
  cat( "Reading ", inputImages[i], "(", i, 
    " out of ", length( inputImages ), ").\n", sep = '' )  
  image <- antsImageRead( inputImages[i] )
  baseId <- basename( inputImages[i] )
  imageChannels <- splitChannels( image )

  # Combine to a single channel for simplicity using Y_{linear}
  # defined at https://en.wikipedia.org/wiki/Grayscale
  
  luminanceImage <- as.antsImage( 
    0.2126 * as.array( imageChannels[[1]] ) +
    0.7152 * as.array( imageChannels[[2]] ) +
    0.0722 * as.array( imageChannels[[3]] ) )

  cat( "  Producing interpolated image for comparison.\n" )

  highResolutionSpacing <- antsGetSpacing( luminanceImage )

  # The low resolution spacing is a factor of 2 because of the specified
  # highRes/lowRes patch sizes.  We first downsample to the low resolution
  # and then upsample for comparison.
  lowResolutionSpacing <- highResolutionSpacing * 2

  luminanceImageLowResolution <- resampleImage( luminanceImage, 
    lowResolutionSpacing, useVoxels = FALSE, interpType = 1 )
  luminanceImageInterpolated <- resampleImage( luminanceImageLowResolution, 
    highResolutionSpacing, useVoxels = FALSE, interpType = 1 )
  
  cat( "  Extracting patches.\n" )
  luminanceImagePatches <- extractImagePatches( luminanceImageInterpolated, 
    highResolutionPatchSize, maxNumberOfPatches = 'all' )
  numberOfPatches <- length( luminanceImagePatches )

  X_test <- array( data = 0, 
    dim = c( numberOfPatches, highResolutionPatchSize, channelSize ) )
  for( j in seq_len( numberOfPatches ) )
    {
    X_test[j,,,1] <- luminanceImagePatches[[j]]
    }

  cat( "  Predicting SR image.\n" )
  predictedData <- srModel %>% predict( X_test, verbose = 1 )

  predictedPatches <- list()
  for( j in seq_len( numberOfPatches ) )
    {
    predictedPatches[[j]] <- predictedData[j,,,1]
    }
  predictedImage <- reconstructImageFromPatches( predictedPatches, luminanceImage )

  # Write output images.
  antsImageWrite( image, 
    paste0( outputDir, baseId, "Original.nii.gz" ) )
  antsImageWrite( luminanceImageInterpolated, 
    paste0( outputDir, baseId, "BsplineInterpolated.nii.gz" ) )
  antsImageWrite( predictedImage, 
    paste0( outputDir, baseId, "SrPredicted.nii.gz" ) )
  }

