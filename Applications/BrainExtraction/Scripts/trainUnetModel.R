library( ANTsR )
library( ANTsRNet )
library( keras )

keras::backend()$clear_session()

baseDirectory <- '/Users/ntustison/Data/BrainExtraction/'
source( paste0( baseDirectory, 'Scripts/unetBatchGenerator.R' ) )

classes <- c( "background", "brain" )

numberOfClassificationLabels <- length( classes )

imageMods <- c( "T1" )
channelSize <- length( imageMods )

dataDirectory <- paste0( baseDirectory, 'Data/' )
brainImageDirectory <- paste0( dataDirectory, 
  'Training/Images/' )
brainImageFiles <- list.files( path = brainImageDirectory, 
  pattern = "*", full.names = TRUE )
brainMaskDirectory <- paste0( dataDirectory, 
  'Training/Masks/' )
brainMaskFiles <-  list.files( path = brainMaskDirectory, 
  pattern = "*", full.names = TRUE )
brainSegmentationDirectory <- paste0( dataDirectory, 
  'Training/Images/' )
brainSegmentationFiles <-  list.files( path = brainSegmentationDirectory, 
  pattern = "*", full.names = TRUE )

templateDirectory <- paste0( dataDirectory, 'Template/' )
reorientTemplateDirectory <- paste0( dataDirectory, 'TemplateReorient/' )
reorientTemplate <- antsImageRead( paste0( templateDirectory, "S_template3_resampled.nii.gz" ), dimension = 3 )
# mniTemplateDirectory <- paste0( dataDirectory, 'mni_icbm152_nlin_sym_09a/' )
# mniTemplate <- antsImageRead( 
#   paste0( mniTemplateDirectory, "mni_icbm152_t1_tal_nlin_sym_09a.nii" ), dimension = 3 )

trainingImageFiles <- list()
trainingSegmentationFiles <- list()
trainingMaskFiles <- list()
trainingTransforms <- list()


cat( "Loading data...\n" )
pb <- txtProgressBar( min = 0, max = length( brainImageFiles ), style = 3 )

count <- 1
for( i in 1:length( brainImageFiles ) )
  {
  trainingImageFiles[[count]] <- brainImageFiles[i]
  trainingMaskFiles[[count]] <- brainMaskFiles[i]

  subjectId <- basename( brainMaskFiles[i] )
  subjectId <- sub( "BrainExtractionMask.nii.gz", '', subjectId )

  xfrmPrefix <- paste0( 'S_', subjectId )
  transformFiles <- list.files( templateDirectory, pattern = xfrmPrefix, full.names = TRUE ) 

  fwdtransforms <- c()
  fwdtransforms[1] <- transformFiles[3]
  fwdtransforms[2] <- transformFiles[1]

  reorientTransform <- paste0( reorientTemplateDirectory, "SR_", subjectId, "0GenericAffine.mat" )

  invtransforms <- c()
  invtransforms[1] <- reorientTransform
  invtransforms[2] <- transformFiles[1]
  invtransforms[3] <- transformFiles[2]

  if( !file.exists( fwdtransforms[1] ) || !file.exists( fwdtransforms[2] ) ||
      !file.exists( invtransforms[1] ) || !file.exists( invtransforms[2] ) ||
      !file.exists( invtransforms[3] ) )
    {
    stop( paste( "Transform", subjectId, "file does not exist.\n" ) )
    }

  trainingTransforms[[count]] <- list( 
    fwdtransforms = fwdtransforms, invtransforms = invtransforms )

  count <- count + 1  
  setTxtProgressBar( pb, i )
  }

###
#
# Create the Unet model
#

resampledImageSize <- dim( reorientTemplate )

unetModel <- createUnetModel3D( c( resampledImageSize, channelSize ), 
  numberOfClassificationLabels = numberOfClassificationLabels, 
  numberOfLayers = 4, numberOfFiltersAtBaseLayer = 16, dropoutRate = 0.0,
  convolutionKernelSize = c( 5, 5, 5 ), deconvolutionKernelSize = c( 5, 5, 5 ) )

unetModel %>% compile( loss = loss_multilabel_dice_coefficient_error,
  optimizer = optimizer_adam( lr = 0.00001 ),  
  metrics = c( multilabel_dice_coefficient ) )

###
#
# Set up the training generator
#

batchSize <- 12L

# Split trainingData into "training" and "validation" componets for
# training the model.

numberOfData <- length( brainImageFiles )
sampleIndices <- sample( numberOfData )

validationSplit <- floor( 0.8 * numberOfData )
trainingIndices <- sampleIndices[1:validationSplit]
numberOfTrainingData <- length( trainingIndices )
validationIndices <- sampleIndices[( validationSplit + 1 ):numberOfData]
numberOfValidationData <- length( validationIndices )

trainingData <- unetImageBatchGenerator$new( 
  imageList = trainingImageFiles[trainingIndices], 
  segmentationList = trainingMaskFiles[trainingIndices], 
  transformList = trainingTransforms[trainingIndices], 
  referenceImageList = trainingImageFiles, 
  referenceTransformList = trainingTransforms
  )

trainingDataGenerator <- trainingData$generate( batchSize = batchSize, 
  resampledImageSize = resampledImageSize, doRandomHistogramMatching = FALSE,
  referenceImage = reorientTemplate )

validationData <- unetImageBatchGenerator$new( 
  imageList = trainingImageFiles[validationIndices], 
  segmentationList = trainingMaskFiles[validationIndices], 
  transformList = trainingTransforms[validationIndices],
  referenceImageList = trainingImageFiles, 
  referenceTransformList = trainingTransforms
  )

validationDataGenerator <- validationData$generate( batchSize = batchSize,
  resampledImageSize = resampledImageSize, doRandomHistogramMatching = FALSE,
  referenceImage = reorientTemplate )

###
#
# Run training
#
track <- unetModel$fit_generator( 
  generator = reticulate::py_iterator( trainingDataGenerator ), 
  steps_per_epoch = ceiling( numberOfTrainingData / batchSize ),
  epochs = 200,
  validation_data = reticulate::py_iterator( validationDataGenerator ),
  validation_steps = ceiling( numberOfValidationData / batchSize ),
  callbacks = list( 
    callback_model_checkpoint( paste0( dataDirectory, "/unetModelWeights.h5" ), 
      monitor = 'val_loss', save_best_only = TRUE, save_weights_only = TRUE,
      verbose = 1, mode = 'auto', period = 1 ),
     callback_reduce_lr_on_plateau( monitor = 'val_loss', factor = 0.1,
       verbose = 1, patience = 10, mode = 'auto' ),
     callback_early_stopping( monitor = 'val_loss', min_delta = 0.001, 
       patience = 20 )
  )
)  


