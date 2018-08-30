library( ANTsR )
library( ANTsRNet )
library( keras )


# Create the training data which consists of randomly selected
# patches.

baseDir <- './'
inputImageDir <- paste0( baseDir, 'TrainingImages/' )

patchesDir <- paste0( baseDir, 'TrainingPatches/' )
patchesHighResolutionDir <- paste0( patchesDir, 'HighResolution/' )
patchesLowResolutionDir <- paste0( patchesDir, 'LowResolution/' )
dir.create( patchesHighResolutionDir, showWarnings = FALSE, recursive = TRUE )
dir.create( patchesLowResolutionDir, showWarnings = FALSE, recursive = TRUE )

inputImages <- list.files( path = inputImageDir, pattern = ".bmp",
  full.names = TRUE, recursive = TRUE )
numberOfInputImages <- length( inputImages )

numberOfPatchesPerImage <- 10000

lowResolutionPatches <- list.files( path = patchesLowResolutionDir, 
  pattern = ".nii.gz", full.names = TRUE )
highResolutionPatches <- list.files( path = patchesHighResolutionDir, 
  pattern = ".nii.gz", full.names = TRUE )

highResolutionPatchSize <- c( 64, 64 )
lowResolutionPatchSize <- c( 32, 32 )

patchCreationIsDone <- FALSE
if( ( length( lowResolutionPatches ) == length( highResolutionPatches ) ) &&
  length( lowResolutionPatches ) == numberOfInputImages * numberOfPatchesPerImage )
  {
  patchCreationIsDone <- TRUE  
  }

if( !patchCreationIsDone )
  {
  cat( "Creating the patches for training.\n" )  
  for( i in seq_len( numberOfInputImages ) )
    {
    cat( "  Reading ", inputImages[i], "(", i, 
      " out of ", length( inputImages ), ").\n", sep = '' )  
    image <- antsImageRead( inputImages[i] )
    imageChannels <- splitChannels( image )

    # Combine to a single channel for simplicity using Y_{linear}
    # defined at https://en.wikipedia.org/wiki/Grayscale
    
    luminanceImage <- as.antsImage( 
      0.2126 * as.array( imageChannels[[1]] ) +
      0.7152 * as.array( imageChannels[[2]] ) +
      0.0722 * as.array( imageChannels[[3]] ) )
    channelSize <- 1                  

    cat( "  Extracting high-resolution patches.\n" )

    highResolutionPatches <- extractImagePatches( luminanceImage, 
      highResolutionPatchSize, maxNumberOfPatches = numberOfPatchesPerImage )
    numberOfPatches <- length( highResolutionPatches )

    cat( "      --> Generating low-resolution patches.\n" )

    pb <- txtProgressBar( min = 0, max = numberOfPatches, style = 3 )

    lowResolutionPatches <- list()
    for( j in seq_len( numberOfPatches ) )
      {
      highResolutionImage <- as.antsImage( highResolutionPatches[[j]] )
      
      # We downsample the high resolution image to create the low resolution
      # image.  We then upsample with b-spline interpolation to match the
      # size of the high resolution patches.  This mimics the training/prediction
      # process.

      lowResolutionImage <- resampleImage( highResolutionImage, 
        lowResolutionPatchSize, useVoxels = TRUE, interpType = 1 )
      lowResolutionImage <- resampleImage( lowResolutionImage, 
        highResolutionPatchSize, useVoxels = TRUE, interpType = 4 )
      lowResolutionPatches[[j]] <- as.array( lowResolutionImage )

      antsImageWrite( lowResolutionImage, 
        paste0( patchesLowResolutionDir, i, "_", j, ".nii.gz" ) )
      antsImageWrite( highResolutionImage, 
        paste0( patchesHighResolutionDir, i, "_", j, ".nii.gz" ) )

      setTxtProgressBar( pb, j )
      }
    cat( "\n")
    }  
  }

###
# 
# Create X_train and Y_train
#

numberOfPatches <- 0.01 * length( highResolutionPatches )
channelSize <- 1                  

X <- array( data = 0, 
  dim = c( numberOfPatches, highResolutionPatchSize, channelSize ) )
Y <- array( data = 0, 
  dim = c( numberOfPatches, highResolutionPatchSize, channelSize ) )

lowResolutionPatches <- list.files( path = patchesLowResolutionDir, 
  pattern = ".nii.gz", full.names = TRUE )
highResolutionPatches <- list.files( path = patchesHighResolutionDir, 
  pattern = ".nii.gz", full.names = TRUE )

cat( "Creating X_train/Y_train.\n" )
pb <- txtProgressBar( min = 0, max = numberOfPatches, style = 3 )
for( i in seq_len( numberOfPatches ) )
  {
  xarray <- as.array( antsImageRead( lowResolutionPatches[i] ) )
  X[i,,,1] <- ( xarray - mean( xarray ) ) / sd( xarray )

  yarray <- as.array( antsImageRead( highResolutionPatches[i] ) )
  Y[i,,,1] <- ( yarray - mean( yarray ) ) / sd( yarray )
  
  setTxtProgressBar( pb, i )
  }
cat( "\n" )  

numberOfTrainingPatches <- floor( 0.8 * numberOfPatches );

X_train <- X[1:numberOfTrainingPatches,,,, drop = FALSE]
Y_train <- Y[1:numberOfTrainingPatches,,,, drop = FALSE]

X_test <- X[( numberOfTrainingPatches + 1 ):numberOfPatches,,,, drop = FALSE]
Y_test <- Y[( numberOfTrainingPatches + 1 ):numberOfPatches,,,, drop = FALSE]

###
#
# Create the super resolution model
#
srModel <- createImageSuperResolutionModel2D( 
 c( highResolutionPatchSize, channelSize ),
 convolutionKernelSizes = list( c( 9, 9 ), c( 1, 1 ), c( 5, 5 ) ) )
srModel %>% compile( loss = loss_mean_squared_error,
  optimizer = optimizer_adam( lr = 0.001 ),  
  metrics = c( 'mse', peak_signal_to_noise_ratio ) )

##
#
# Train
#
srModel %>% fit( X_train, Y_train, batch_size = 32, epochs = 200,
  callbacks = list( 
    callback_model_checkpoint( paste0( baseDir, "srWeights.h5" ), 
      monitor = 'val_loss', save_best_only = TRUE, save_weights_only = TRUE,
      verbose = 1, mode = 'auto', period = 1 ),
     callback_reduce_lr_on_plateau( monitor = 'val_loss', factor = 0.1,
       verbose = 1, patience = 10, mode = 'auto' ),
     callback_early_stopping( monitor = 'val_loss', min_delta = 0.001, 
       patience = 10 )
    ),
  validation_data = list( X_test, Y_test )  
  )
