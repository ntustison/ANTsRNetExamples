library( ANTsR )
library( ANTsRNet )
library( keras )

baseDir <- '../'
inputImageDir <- paste0( baseDir, 'TrainingImages/' )

image <- antsImageRead( "./TrainingImages/t1.bmp" )
imageChannels <- splitChannels( image )
channelSize <- length( imageChannels )

highResolutionPatchSize <- c( 64, 64 )
lowResolutionPatchSize <- c( 32, 32 )

cat( "Extracting high-resolution patches.\n" )

highResolutionPatches <- list()
for( i in seq_len( channelSize ) )
  {
  highResolutionPatches[[i]] <- extractImagePatches( 
    imageChannels[[i]], highResolutionPatchSize, 'all' )
  }

numberOfPatches <- length( highResolutionPatches[[1]] )

lowResolutionPatches <- list()
for( i in seq_len( channelSize ) )
  {
  cat( "Generating low-resolution patches for channel ", i, ".\n", sep = '' )

  pb <- txtProgressBar( min = 0, max = numberOfPatches, 
    title = paste0( "Channel", i ), style = 3 )

  lowResolutionPatchesPerChannels <- list()
  for( j in seq_len( numberOfPatches ) )
    {
    highResolutionImage <- as.antsImage( highResolutionPatches[[i]][[j]] )
    lowResolutionImage <- resampleImage( highResolutionImage, 
      lowResolutionPatchSize, useVoxels = TRUE, interpType = 4 )
    lowResolutionPatchesPerChannels[[j]] <- as.array( lowResolutionImage )

    setTxtProgressBar( pb, j )
    }
  cat( "\n")

  lowResolutionPatches[[i]] <- lowResolutionPatchesPerChannels
  }

srModel <- createImageSuperResolutionModel2D( 
 c( lowResolutionPatchSize, channelSize ),
 convolutionKernelSizes = list( c( 9, 9 ), c( 1, 1 ), c( 5, 5 ) ) )
srModel %>% compile( loss = 'mse',
  optimizer = optimizer_adam( lr = 0.00001 ),  
  metrics = c( peak_signal_to_noise_ratio ) )


