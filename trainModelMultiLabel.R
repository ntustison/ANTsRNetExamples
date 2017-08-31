library( ANTsR )
library( keras )
library( abind )
library( ggplot2 )

baseDirectory <- './'
dataDirectory <- paste0( baseDirectory, 'Images/' )
trainingDirectory <- paste0( dataDirectory, 'TrainingData/' )

source( paste0( baseDirectory, 'createUnetModel.R' ) )

trainingImageFiles <- list.files( path = trainingDirectory, pattern = "H1_2D", full.names = TRUE )
trainingMaskFiles <- list.files( path = trainingDirectory, pattern = "Mask_2D", full.names = TRUE )

trainingImages <- list()
trainingMasks <- list()
trainingImageArrays <- list()
trainingMaskArrays <- list()

for ( i in 1:length( trainingImageFiles ) )
  {
  trainingImages[[i]] <- antsImageRead( trainingImageFiles[i], dimension = 2 )    
  trainingMasks[[i]] <- antsImageRead( trainingMaskFiles[i], dimension = 2 )    

  trainingImageArrays[[i]] <- as.array( trainingImages[[i]] )
  trainingMaskArrays[[i]] <- as.array( trainingMasks[[i]] )  
  # trainingMaskArrays[[i]][which( trainingMaskArrays[[i]] > 1 )] <- 1
  }

trainingData <- abind( trainingImageArrays, along = 3 )  
trainingData <- aperm( trainingData, c( 3, 1, 2 ) )
trainingData <- ( trainingData - mean( trainingData ) ) / sd( trainingData )

X_train <- array( trainingData, dim = c( dim( trainingData ), 1 ) )

trainingLabelData <- abind( trainingMaskArrays, along = 3 )  
trainingLabelData <- aperm( trainingLabelData, c( 3, 1, 2 ) )

segmentationLabels <- sort( unique( as.vector( trainingLabelData ) ) )
numberOfLabels <- length( segmentationLabels )

cat( "Segmentation with ", numberOfLabels, " labels: ", segmentationLabels, ".\n", sep = "" )

# Different implementation of keras::to_categorical().  The ordering 
# of the array elements seems to get screwed up.

Y_train <- trainingLabelData
Y_train[which( trainingLabelData == 0)] <- 1
Y_train[which( trainingLabelData != 0)] <- 0

for( i in 2:numberOfLabels )
  {
  Y_train_label <- trainingLabelData 
  Y_train_label[which( trainingLabelData == segmentationLabels[i] )] <- 1
  Y_train_label[which( trainingLabelData != segmentationLabels[i] )] <- 0

  Y_train <- abind( Y_train, Y_train_label, along = 4 )
  }

unetModel <- createUnetModel2D( c( dim( trainingImageArrays[[1]] ), 1 ), numberOfClassificationLabels = numberOfLabels, layers = 1:5 )
track <- unetModel %>% fit( X_train, Y_train, 
                 epochs = 50, batch_size = 32, verbose = 1, shuffle = TRUE,
                 callbacks = list( 
                   callback_model_checkpoint( paste0( baseDirectory, "weightsMultiLabel.h5" ), monitor = 'val_loss', save_best_only = TRUE )
                 #  callback_early_stopping( patience = 2, monitor = 'loss' ),
                  #  callback_reduce_lr_on_plateau( monitor = "val_loss", factor = 0.1 )
                 ), 
                 validation_split = 0.2 )
## Save the model

save_model_weights_hdf5( unetModel, filepath = paste0( baseDirectory, 'unetModelMultiLabelWeights.h5' ) )
save_model_hdf5( unetModel, filepath = paste0( baseDirectory, 'unetModelMultiLabel.h5' ), overwrite = TRUE )

## Plot the model fitting

epochs <- 1:length( track$metrics$loss )

unetModelDataFrame <- data.frame( Epoch = rep( epochs, 2 ), 
                                  Type = c( rep( 'Training', length( epochs ) ), rep( 'Validation', length( epochs ) ) ),
                                  Loss =c( track$metrics$loss, track$metrics$val_loss ), 
                                  Accuracy = c( track$metrics$dice_coefficient, track$metrics$val_dice_coefficient )
                                )

unetModelLossPlot <- ggplot( data = unetModelDataFrame, aes( x = Epoch, y = Loss, colour = Type ) ) +
                 geom_point( shape = 1, size = 0.5 ) +
                 geom_line( size = 0.3 ) +
                 ggtitle( "Loss" )
                

unetModelAccuracyPlot <- ggplot( data = unetModelDataFrame, aes( x = Epoch, y = Accuracy, colour = Type ) ) +
                 geom_point( shape = 1, size = 0.5 ) +
                 geom_line( size = 0.3 ) +
                 ggtitle( "Accuracy")

ggsave( paste0( baseDirectory, "unetModelLossPlot.pdf" ), plot = unetModelLossPlot, width = 5, height = 2, units = 'in' )
ggsave( paste0( baseDirectory, "unetModelAccuracyPlot.pdf" ), plot = unetModelAccuracyPlot, width = 5, height = 2, units = 'in' )









