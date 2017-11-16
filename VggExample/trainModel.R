library( ANTsR )
library( keras )
library( abind )
library( ggplot2 )

# Dog vs. cat data available from here:
#    https://www.kaggle.com/c/dogs-vs-cats/data

baseDirectory <- './'
dataDirectory <- paste0( baseDirectory, 'Images/' )
modelDirectory <- paste0( baseDirectory, '../Models/' )
trainingDirectory <- paste0( dataDirectory, 'TrainingData/' )

source( paste0( modelDirectory, 'createVggModel.R' ) )

trainingImageFiles <- list.files( 
  path = trainingDirectory, pattern = "*.jpg", full.names = TRUE )

trainingProportion <- 0.5
set.seed( 1234 )
trainingIndices <- sample.int( 
  length( trainingImageFiles ), size = length( trainingImageFiles ) * trainingProportion )
trainingClassifications <- rep( 0, length( trainingIndices ) )

trainingImageSize <- c( 224, 224 )

trainingImages <- list()
trainingImageArrays <- list()

for ( i in 1:length( trainingIndices ) )
  {
  cat( "Reading ", trainingImageFiles[trainingIndices[i]], "\n" )
  trainingImages[[i]] <- resampleImage( 
    antsImageRead( trainingImageFiles[trainingIndices[i]], dimension = 2 ),
    trainingImageSize, useVoxels = TRUE )
  trainingImageArrays[[i]] <- as.array( trainingImages[[i]] )
  if( grepl( "dog", trainingImageFiles[trainingIndices[i]] ) )
    {
    trainingClassifications[i] <- 1
    }
  }

trainingData <- abind( trainingImageArrays, along = 3 )  
trainingData <- aperm( trainingData, c( 3, 1, 2 ) )
trainingData <- ( trainingData - mean( trainingData ) ) / sd( trainingData )

X_train <- array( trainingData, dim = c( dim( trainingData ), 1 ) )

segmentationLabels <- sort( unique( trainingClassifications ) )
numberOfLabels <- length( segmentationLabels )
Y_train <- to_categorical( trainingClassifications, numberOfLabels )

vggModel <- createVggModel2D( c( dim( trainingImageArrays[[1]] ), 1 ), 
  numberOfClassificationLabels = numberOfLabels )

track <- vggModel %>% fit( X_train, Y_train, 
                 epochs = 40, batch_size = 32, verbose = 1, shuffle = TRUE,
                 callbacks = list( 
                   callback_model_checkpoint( paste0( baseDirectory, "vggWeights.h5" ),
                     monitor = 'val_loss', save_best_only = TRUE )
                  # callback_early_stopping( patience = 2, monitor = 'loss' ),
                  #  callback_reduce_lr_on_plateau( monitor = "val_loss", factor = 0.1 )
                 ), 
                 validation_split = 0.2 )
# Save the model

save_model_weights_hdf5( 
  vggModel, filepath = paste0( baseDirectory, 'vggWeights.h5' ) )
save_model_hdf5( 
  vggModel, filepath = paste0( baseDirectory, 'vggModel.h5' ), overwrite = TRUE )

## Plot the model fitting

epochs <- 1:length( track$metrics$loss )

vggModelDataFrame <- data.frame( Epoch = rep( epochs, 2 ), 
                                  Type = c( rep( 'Training', length( epochs ) ), rep( 'Validation', length( epochs ) ) ),
                                  Loss =c( track$metrics$loss, track$metrics$val_loss ), 
                                  Accuracy = c( track$metrics$multilabel_dice_coefficient, track$metrics$val_multilabel_dice_coefficient )
                                )

vggModelLossPlot <- ggplot( data = vggModelDataFrame, aes( x = Epoch, y = Loss, colour = Type ) ) +
                 geom_point( shape = 1, size = 0.5 ) +
                 geom_line( size = 0.3 ) +
                 ggtitle( "Loss" )
                

vggModelAccuracyPlot <- ggplot( data = vggModelDataFrame, aes( x = Epoch, y = Accuracy, colour = Type ) ) +
                 geom_point( shape = 1, size = 0.5 ) +
                 geom_line( size = 0.3 ) +
                 ggtitle( "Accuracy")

ggsave( paste0( baseDirectory, "vggModelLossPlot.pdf" ), plot = vggModelLossPlot, width = 5, height = 2, units = 'in' )
ggsave( paste0( baseDirectory, "vggModelAccuracyPlot.pdf" ), plot = vggModelAccuracyPlot, width = 5, height = 2, units = 'in' )









