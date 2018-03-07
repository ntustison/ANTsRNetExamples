library( ANTsR )
library( keras )
library( abind )
library( ggplot2 )

baseDirectory <- './'
dataDirectory <- paste0( baseDirectory, 'Images/' )
modelDirectory <- paste0( baseDirectory, '../../Models/' )

source( paste0( modelDirectory, 'createUnetModel.R' ) )
source( paste0( baseDirectory, 'unetBatchGenerator.R' ) )


trainingImageDirectory <- paste0( dataDirectory, 'TrainingData/' )
trainingImageFiles <- list.files( 
  path = trainingImageDirectory, pattern = "H1_2D", full.names = TRUE )
trainingMaskFiles <- list.files( 
  path = trainingImageDirectory, pattern = "Mask_2D", full.names = TRUE )

trainingTransformDirectory <- paste0( dataDirectory, 'TemplateTransforms/' )

trainingTransforms <- list()
trainingImages <- list()
trainingSegmentations <- list()

for( i in 1:length( trainingImageFiles ) )
  {
  trainingImages[[i]] <- antsImageRead( trainingImageFiles[i], dimension = 2 )
  trainingSegmentations[[i]] <- antsImageRead( trainingMaskFiles[i], dimension = 2 )

  id <- basename( trainingImageFiles[i] ) 
  id <- gsub( "H1_2D.nii.gz", '', id )

  xfrmPrefix <- paste0( trainingTransformDirectory, 'T_', id, i - 1 )

  fwdtransforms <- list()
  fwdtransforms[[1]] <- paste0( xfrmPrefix, 'Warp.nii.gz' )
  fwdtransforms[[2]] <- paste0( xfrmPrefix, 'Affine.txt' )
  invtransforms <- list()
  invtransforms[[1]] <- paste0( xfrmPrefix, 'Affine.txt' )
  invtransforms[[2]] <- paste0( xfrmPrefix, 'InverseWarp.nii.gz' )

  trainingTransforms[[i]] <- list( 
    fwdtransforms = fwdtransforms, invtransforms = invtransforms )
  }

unetModel <- createUnetModel2D( c( dim( trainingImages[[1]] ), 1 ), 
  numberOfClassificationLabels = numberOfLabels, layers = 1:4 )

unetModel %>% compile( loss = loss_multilabel_dice_coefficient_error,
  optimizer = optimizer_adam( lr = 0.0001 ),  
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

validationSplit <- 40
trainingIndices <- sampleIndices[1:validationSplit]
validationIndices <- sampleIndices[( validationSplit + 1 ):numberOfTrainingData]

trainingData <- unetImageBatchGenerator$new( 
  imageList = trainingImages[trainingIndices], 
  segmentationList = trainingSegmentations[trainingIndices], 
  transformList = trainingTransforms[trainingIndices] )

trainingDataGenerator <- trainingData$generate( batchSize = batchSize )

validationData <- unetImageBatchGenerator$new( 
  imageList = trainingImages[validationIndices], 
  segmentationList = trainingSegmentations[validationIndices], 
  transformList = trainingTransforms[validationIndices] )

validationDataGenerator <- trainingData$generate( batchSize = batchSize )

###
#
# Run training
#

track <- unetModel$fit_generator( 
  generator = reticulate::py_iterator( trainingDataGenerator ), 
  steps_per_epoch = ceiling( length( trainingIndices ) / batchSize ),
  epochs = 100,
  validation_data = reticulate::py_iterator( validationDataGenerator ),
  validation_steps = ceiling( length( validationIndices ) / batchSize ),
  callbacks = list( 
    callback_model_checkpoint( paste0( baseDirectory, "ssd7Weights.h5" ), 
      monitor = 'val_loss', save_best_only = TRUE, save_weights_only = TRUE,
      verbose = 1, mode = 'auto', period = 1 ),
    callback_early_stopping( monitor = 'val_loss', min_delta = 0.001, 
      patience = 10 ),
    callback_reduce_lr_on_plateau( monitor = 'val_loss', factor = 0.5,
      patience = 0, epsilon = 0.001, cooldown = 0 )
                  # callback_early_stopping( patience = 2, monitor = 'loss' ),
    )
  )







