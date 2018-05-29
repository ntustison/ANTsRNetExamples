library( ANTsR )
library( ANTsRNet )
library( keras )
library( ggplot2 )

keras::backend()$clear_session()

baseDirectory <- './'
dataDirectory <- paste0( baseDirectory, 'Images/Proton/' )

source( paste0( baseDirectory, 'unetProtonBatchGenerator.R' ) )

trainingImageDirectory <- paste0( dataDirectory, 'TrainingData/' )
trainingImageFiles <- list.files( 
  path = trainingImageDirectory, pattern = "N4Denoised_2D", full.names = TRUE )

trainingTransformDirectory <- paste0( dataDirectory, 'TemplateTransforms/' )

trainingTransforms <- list()
trainingImages <- list()
trainingSegmentations <- list()

for( i in 1:length( trainingImageFiles ) )
  {
  trainingImages[[i]] <- antsImageRead( trainingImageFiles[i], dimension = 2 )
  
  id <- basename( trainingImageFiles[i] ) 
  id <- gsub( "N4Denoised_2D.nii.gz", '', id )

  trainingSegmentationFile <- paste0( trainingImageDirectory, id, "Mask_2D.nii.gz" )
  trainingSegmentations[[i]] <- antsImageRead( trainingSegmentationFile, dimension = 2 )

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
  convolutionKernelSize = c( 5, 5 ), deconvolutionKernelSize = c( 5, 5 ),
  numberOfClassificationLabels = 3, numberOfLayers = 4 )

# # multilabel Dice loss function
# unetModel %>% compile( loss = loss_multilabel_dice_coefficient_error,
#   optimizer = optimizer_adam( lr = 0.0001 ),  
#   metrics = c( multilabel_dice_coefficient ) )

# categorical cross entropy loss function
unetModel %>% compile( loss = "categorical_crossentropy",
  optimizer = optimizer_adam( lr = 0.0001 ),  
  metrics = c( "acc" ) )

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
  steps_per_epoch = ceiling( 400 / batchSize ),
  epochs = 200,
  validation_data = reticulate::py_iterator( validationDataGenerator ),
  validation_steps = ceiling( 200 / batchSize ),
  callbacks = list( 
    callback_model_checkpoint( paste0( baseDirectory, "unetProtonWeights.h5" ), 
      monitor = 'val_loss', save_best_only = TRUE, save_weights_only = TRUE,
      verbose = 1, mode = 'auto', period = 1 ),
     callback_reduce_lr_on_plateau( monitor = 'val_loss', factor = 0.1,
       verbose = 1, patience = 10, mode = 'auto' ),
     callback_early_stopping( monitor = 'val_loss', min_delta = 0.001, 
       patience = 10 )
    )
  )

## Plot the model fitting

epochs <- 1:length( track$metrics$loss )

unetModelDataFrame <- data.frame( 
  Epoch = rep( epochs, 2 ), 
  Type = c( rep( 'Training', length( epochs ) ), rep( 'Validation', length( epochs ) ) ),
  Loss =c( track$metrics$loss, track$metrics$val_loss ), 
  Accuracy = c( track$metrics$multilabel_dice_coefficient, 
                track$metrics$val_multilabel_dice_coefficient )
  )

unetModelLossPlot <- 
  ggplot( data = unetModelDataFrame, aes( x = Epoch, y = Loss, colour = Type ) ) +
  geom_point( shape = 1, size = 0.5 ) +
  geom_line( size = 0.3 ) +
  ggtitle( "Loss" )

unetModelAccuracyPlot <- 
  ggplot( data = unetModelDataFrame, aes( x = Epoch, y = Accuracy, colour = Type ) ) +
  geom_point( shape = 1, size = 0.5 ) +
  geom_line( size = 0.3 ) +
  ggtitle( "Accuracy")

ggsave( paste0( baseDirectory, "unetModelLossPlot.pdf" ), 
  plot = unetModelLossPlot, width = 5, height = 2, units = 'in' )
ggsave( paste0( baseDirectory, "unetModelAccuracyPlot.pdf" ), 
  plot = unetModelAccuracyPlot, width = 5, height = 2, units = 'in' )







