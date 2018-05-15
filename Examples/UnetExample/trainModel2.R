library( ANTsR )
library( ANTsRNet )
library( keras )

keras::backend()$clear_session()

baseDirectory <- './'
dataDirectory <- paste0( baseDirectory, 'Images/' )

source( paste0( baseDirectory, 'unetBatchGenerator.R' ) )

trainingImageDirectory <- paste0( dataDirectory, 'TrainingData/' )
trainingImageFiles <- list.files( 
  path = trainingImageDirectory, pattern = "N4Denoised_2D", full.names = TRUE )
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
  id <- gsub( "N4Denoised_2D.nii.gz", '', id )

  xfrmPrefix <- paste0( trainingTransformDirectory, 'T_', id, i - 1 )

  fwdtransforms <- c()
  fwdtransforms[1] <- paste0( xfrmPrefix, 'Warp.nii.gz' )
  fwdtransforms[2] <- paste0( xfrmPrefix, 'Affine.txt' )
  invtransforms <- c()
  invtransforms[1] <- paste0( xfrmPrefix, 'Affine.txt' )
  invtransforms[2] <- paste0( xfrmPrefix, 'InverseWarp.nii.gz' )

  trainingTransforms[[i]] <- list( 
    fwdtransforms = fwdtransforms, invtransforms = invtransforms )
  }

unetModel <- createUnetModel2D( c( dim( trainingImages[[1]] ), 1 ), 
  numberOfClassificationLabels = 3, convolutionKernelSize = c( 5, 5 ),
  deconvolutionKernelSize = c( 5, 5 ), lowestResolution = 32,
  dropoutRate = 0.2 )

unetModel %>% compile( loss = loss_multilabel_dice_coefficient_error,
  optimizer = optimizer_adam( lr = 0.0001 ),  
  metrics = c( multilabel_dice_coefficient ) )

###
#
# Set up the training generator
#
batchSize <- 20L

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
  transformList = trainingTransforms[trainingIndices], 
  referenceImageList = trainingImages, 
  referenceTransformList = trainingTransforms
  )

trainingDataGenerator <- trainingData$generate( batchSize = batchSize )

validationData <- unetImageBatchGenerator$new( 
  imageList = trainingImages[validationIndices], 
  segmentationList = trainingSegmentations[validationIndices], 
  transformList = trainingTransforms[validationIndices],
  referenceImageList = trainingImages, 
  referenceTransformList = trainingTransforms
  )

validationDataGenerator <- validationData$generate( batchSize = batchSize )

###
#
# Run training
#

track <- unetModel$fit_generator( 
  generator = reticulate::py_iterator( trainingDataGenerator ), 
#  steps_per_epoch = ceiling( 400 / batchSize ),
  steps_per_epoch = ceiling( 400 / batchSize ),
  epochs = 200,
  validation_data = reticulate::py_iterator( validationDataGenerator ),
  validation_steps = ceiling( 200 / batchSize ),
  callbacks = list( 
    callback_model_checkpoint( paste0( baseDirectory, "unetWeights.h5" ), 
      monitor = 'val_loss', save_best_only = TRUE, save_weights_only = TRUE,
      verbose = 1, mode = 'auto', period = 1 ),
     callback_reduce_lr_on_plateau( monitor = 'val_loss', factor = 0.1,
       verbose = 1, patience = 10, mode = 'auto' )
      # ,
    #  callback_early_stopping( monitor = 'val_loss', min_delta = 0.001, 
    #    patience = 10 ),
    )
  )







