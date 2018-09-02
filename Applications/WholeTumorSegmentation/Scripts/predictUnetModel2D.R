library( ANTsR )
library( ANTsRNet )
library( keras )

keras::backend()$clear_session()

baseDirectory <- '../'
dataDirectory <- paste0( baseDirectory, 'Data/' )

classes <- c( "background", "tumor" )
numberOfClassificationLabels <- length( classes )

imageMods <- c( "T1xFLAIRWarped", "T1xCONTRASTWarped" )
channelSize <- length( imageMods )

subjectDirs <- c( 
  list.dirs( path = paste0( dataDirectory, 'Images/' ),
    full.names = TRUE, recursive = FALSE  ) )

templateDirectory <- paste0( dataDirectory, 'Template/' )
reorientTemplate <- antsImageRead( paste0( templateDirectory, 
  "S_template3.nii.gz" ), dimension = 3 )

testingImageFiles <- list()
testingMaskFiles <- list()
testingReorientationTransforms <- list()

count <- 1
for( i in 1:length( subjectDirs ) )
  {
  filesExist <- TRUE

  subjectTestingImageFiles <- c()
  for( j in 1:channelSize )
    {
    subjectTestingImageFiles[j] <- 
      paste0( subjectDirs[i], "/", imageMods[j], ".nii.gz" )
    if( ! file.exists( subjectTestingImageFiles[j] ) )  
      {
      filesExist <- FALSE  
      }
    }
  if( filesExist )
    {
    testingImageFiles[[count]] <- subjectTestingImageFiles
    testingReorientationTransforms[[count]] <- 
      paste0( subjectDirs[i], "/TR_0GenericAffine.mat" )
    testingMaskFiles[[count]] <- 
      paste0( subjectDirs[i], "/BrainMaskProbability.nii.gz" )
    count <- count + 1
    }
  }

###
#
# Create the Unet model
#

sliceRadius <- 0 
direction <- 1
resampledSliceSize <- dim( reorientTemplate )[-direction]

unetModel <- createUnetModel2D( c( resampledSliceSize, channelSize * ( 2 * sliceRadius + 1 ) ), 
numberOfClassificationLabels = length( classes ), 
  numberOfLayers = 4, numberOfFiltersAtBaseLayer = 32, dropoutRate = 0.0,
  convolutionKernelSize = c( 5, 5 ), deconvolutionKernelSize = c( 5, 5 ) )

load_model_weights_hdf5( unetModel, 
  filepath = paste0( baseDirectory, 'unetModelWeights2D.h5' ) )

unetModel %>% compile( loss = "categorical_crossentropy",
 optimizer = optimizer_adam( lr = 0.0001 ),
 metrics = c( "acc", multilabel_dice_coefficient ) )

batchSize <- length( testingImageFiles )

for( i in 1:length( testingImageFiles ) )
  {
  if( !file.exists( testingMaskFiles[[i]] ) )
    {
    next;
    }

  cat( "Reading", dirname( testingImageFiles[[i]][1] ), "\n" )  
  subjectDirectory <- dirname( testingImageFiles[[i]][1] )

  mask <- antsImageRead( testingMaskFiles[[i]], dimension = 3 )
  mask <- thresholdImage( mask, 0.5, 10.0, 1.0, 0.0 )

  X_test <- array( data = 0, dim = c( dim( reorientTemplate )[direction], 
    dim( reorientTemplate )[-direction], channelSize ) )

  numberOfSlices <- 0
  originalSliceSize <- 0
  resampledSliceSize <- 0

  cat( "  Reorienting individual channel images to the template.\n" )
  for( j in 1:channelSize )
    {
    image <- antsImageRead( testingImageFiles[[i]][j], dimension = 3 )
    image <- image * mask

    imageReoriented <- antsApplyTransforms( reorientTemplate, image,
            interpolator = "linear", 
            transformlist = c( testingReorientationTransforms[[i]] ),
            whichtoinvert = c( FALSE )  )

    subjectDirectory <- dirname( testingImageFiles[[i]][1] )
    imageArray <- as.array( imageReoriented )
    imageArray <- ( imageArray - mean( imageArray ) ) / sd( imageArray )
    imageReoriented <- as.antsImage( imageArray, reference = imageReoriented )

    originalSliceSize <- dim( image )[-direction]
    resampledSliceSize <- dim( imageReoriented )[-direction]
  
    numberOfSlices <- dim( imageReoriented )[direction]
    for( k in seq_len( numberOfSlices ) )
      {
      imageSlice <- extractSlice( imageReoriented, k, direction )  
      X_test[k,,,j] <- as.array( imageSlice )
      }
    }

  cat( "  Doing prediction.\n" )
  predictedData <- unetModel %>% predict( X_test, verbose = 1 )
  probabilitySlices <- decodeUnet( predictedData, imageSlice )

  cat( "  Creating probability mask.\n" )
  probabilityArray <- array( data = 0, dim = dim( reorientTemplate ) )
  for( k in seq_len( numberOfSlices ) )  
    {
    probabilitySlice <- probabilitySlices[[k]][[2]]  
    probabilityArray[k,,] <- as.array( probabilitySlice )
    }
  probabilityImageReoriented <- as.antsImage( probabilityArray, reference = reorientTemplate )
  probabilityImage <- antsApplyTransforms( image, probabilityImageReoriented,
    interpolator = "linear",   
    transformlist = c( testingReorientationTransforms[[i]] ),
    whichtoinvert = c( TRUE )  )
  imageFileName <- paste0( subjectDirectory, "/TumorMaskProbability.nii.gz" )

  cat( "Writing", imageFileName, "\n" )  
  antsImageWrite( probabilityImage, imageFileName )
  }



