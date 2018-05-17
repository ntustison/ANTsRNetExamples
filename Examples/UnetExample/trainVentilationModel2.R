library( ANTsR )
library( ANTsRNet )
library( keras )

keras::backend()$clear_session()

baseDirectory <- './'
dataDirectory <- paste0( baseDirectory, 'Images/Ventilation/' )

source( paste0( baseDirectory, 'unetVentilationBatchGenerator.R' ) )

classes <- c( "background", "ventilation defect", "hypo-ventilation", "normal ventilation", "hyper-normal ventilation" )
numberOfClassificationLabels <- length( classes )

imageMods <- c( "Ventilation", "Foreground mask" )
channelSize <- length( imageMods )

trainingImageDirectory <- paste0( dataDirectory, 'TrainingData/' )
trainingImageFiles <- list.files( 
  path = trainingImageDirectory, pattern = "Ventilation", full.names = TRUE )
trainingMaskFiles <- list.files( 
  path = trainingImageDirectory, pattern = "Mask", full.names = TRUE )
trainingSegmentationFiles <- list.files( 
  path = trainingImageDirectory, pattern = "Segmentation", full.names = TRUE )

trainingTransformDirectory <- paste0( dataDirectory, 'TemplateTransforms/' )

trainingTransforms <- list()
trainingImages <- list()
trainingSegmentations <- list()

for( i in 1:length( trainingImageFiles ) )
  {
  trainingImages[[i]] <- c( trainingImageFiles[i], trainingMaskFiles[i] )
  trainingSegmentations[[i]] <- trainingSegmentationFiles[i]

  id <- basename( trainingImageFiles[i] ) 
  id <- gsub( "Ventilation.nii.gz", '', id )

  xfrmPrefix <- paste0( trainingTransformDirectory, 'T_', id )

  fwdtransforms <- c()
  fwdtransforms[1] <- paste0( xfrmPrefix, 'Warp.nii.gz' )
  fwdtransforms[2] <- paste0( xfrmPrefix, 'Affine.txt' )
  invtransforms <- c()
  invtransforms[1] <- paste0( xfrmPrefix, 'Affine.txt' )
  invtransforms[2] <- paste0( xfrmPrefix, 'InverseWarp.nii.gz' )

  if( !file.exists( fwdtransforms[1] ) || !file.exists( fwdtransforms[2] ) ||
      !file.exists( invtransforms[1] ) || !file.exists( invtransforms[2] ) )
    {
    stop( "Transform file does not exist.\n" )
    }

  trainingTransforms[[i]] <- list( 
    fwdtransforms = fwdtransforms, invtransforms = invtransforms )
  }

resampledImageSize <- c( 80, 128 )

unetModel <- createUnetModel2D( c( resampledImageSize, channelSize ), 
  numberOfClassificationLabels = numberOfClassificationLabels, 
  convolutionKernelSize = c( 5, 5 ),
  deconvolutionKernelSize = c( 5, 5 ), lowestResolution = 32,
  dropoutRate = 0.2 )

unetModel %>% compile( loss = loss_multilabel_dice_coefficient_error,
  optimizer = optimizer_adam( lr = 0.00001 ),  
  metrics = c( multilabel_dice_coefficient ) )

###
#
# Set up the training generator
#
batchSize <- 32L

# Split trainingData into "training" and "validation" componets for
# training the model.

numberOfTrainingData <- length( trainingImageFiles )

sampleIndices <- sample( numberOfTrainingData )

validationSplit <- 0.8
trainingIndices <- sampleIndices[1:ceiling( validationSplit * numberOfTrainingData )]
validationIndices <- sampleIndices[( ceiling( validationSplit * numberOfTrainingData ) + 1 ):numberOfTrainingData]

trainingData <- unetImageBatchGenerator$new( 
  imageList = trainingImages[trainingIndices], 
  segmentationList = trainingSegmentations[trainingIndices], 
  transformList = trainingTransforms[trainingIndices], 
  referenceImageList = trainingImages, 
  referenceTransformList = trainingTransforms
  )

trainingDataGenerator <- trainingData$generate( batchSize = batchSize, 
  resampledImageSize = resampledImageSize )

validationData <- unetImageBatchGenerator$new( 
  imageList = trainingImages[validationIndices], 
  segmentationList = trainingSegmentations[validationIndices], 
  transformList = trainingTransforms[validationIndices],
  referenceImageList = trainingImages, 
  referenceTransformList = trainingTransforms
  )

validationDataGenerator <- validationData$generate( batchSize = batchSize,
  resampledImageSize = resampledImageSize )

###
#
# Run training
#

track <- unetModel$fit_generator( 
  generator = reticulate::py_iterator( trainingDataGenerator ), 
  steps_per_epoch = ceiling( 5 * length( trainingIndices ) / batchSize ),
  epochs = 200,
  validation_data = reticulate::py_iterator( validationDataGenerator ),
  validation_steps = ceiling( 5 * length( validationIndices ) / batchSize ),
  callbacks = list( 
    callback_model_checkpoint( paste0( baseDirectory, "unetVentilationWeights.h5" ), 
      monitor = 'val_loss', save_best_only = TRUE, save_weights_only = TRUE,
      verbose = 1, mode = 'auto', period = 1 ),
     callback_reduce_lr_on_plateau( monitor = 'val_loss', factor = 0.5,
       verbose = 1, patience = 10, mode = 'auto' )
      # ,
    #  callback_early_stopping( monitor = 'val_loss', min_delta = 0.001, 
    #    patience = 10 ),
    )
  )







