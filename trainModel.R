library( ANTsR )
library( keras )
library( abind )
library( ggplot2 )

baseDirectory <- './'
dataDirectory <- paste0( baseDirectory, 'Images/' )
trainingDirectory <- paste0( dataDirectory, 'TrainingData/' )

source( paste0( baseDirectory, 'createUnetModel.R' ) )

numberOfLabels <- 1

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
  trainingMaskArrays[[i]][which( trainingMaskArrays[[i]] > 1 )] <- 1
  }

trainingData <- abind( trainingImageArrays, along = 3 )  
trainingData <- aperm( trainingData, c( 3, 1, 2 ) )
trainingData <- ( trainingData - mean( trainingData ) ) / sd( trainingData )

trainingLabelData <- abind( trainingMaskArrays, along = 3 )  
trainingLabelData <- aperm( trainingLabelData, c( 3, 1, 2 ) )

X_train <- array( trainingData, dim = c( dim( trainingData ), numberOfLabels ) )
Y_train <- array( trainingLabelData, dim = c( dim( trainingData ), numberOfLabels ) )

unetModel <- createUnetModel2D( dim( trainingImageArrays[[1]] ), numberOfClassificationLabels = numberOfLabels, layers = 1:4 )
track <- unetModel %>% fit( X_train, Y_train, 
                 epochs = 100, batch_size = 32, verbose = 1, shuffle = TRUE,
                 callbacks = list( 
                   callback_model_checkpoint( paste0( baseDirectory, "weights.h5" ), monitor = 'val_loss', save_best_only = TRUE ),
                 #  callback_early_stopping( patience = 2, monitor = 'loss' ),
                   callback_reduce_lr_on_plateau( monitor = "val_loss", factor = 0.1 )
                 ), 
                 validation_split = 0.2 )
## Save the model

save_model_weights_hdf5( unetModel, filepath = paste0( baseDirectory, 'unetModelWeights.h5' ) )
save_model_hdf5( unetModel, filepath = paste0( baseDirectory, 'unetModel.h5' ), overwrite = TRUE )

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









