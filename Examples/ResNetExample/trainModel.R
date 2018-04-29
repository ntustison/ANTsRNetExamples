library( ANTsR )
library( ANTsRNet )
library( keras )
library( abind )
library( ggplot2 )
library( jpeg )

# Dog vs. cat data available from here:
#    https://www.kaggle.com/c/dogs-vs-cats/data
# Also use the human faces from:
#    http://www.vision.caltech.edu/Image_Datasets/Caltech_10K_WebFaces/

trainingProportion <- 0.2
trainingImageSize <- c( 224, 224 )

baseDirectory <- './'
dataDirectory <- paste0( baseDirectory, './Images/' )

trainingDirectories <- c()
trainingDirectories[1] <- paste0( dataDirectory, 'TrainingDataPlanes/' )
trainingDirectories[2] <- paste0( dataDirectory, 'TrainingDataHuman/' )
trainingDirectories[3] <- paste0( dataDirectory, 'TrainingDataCat/' )
trainingDirectories[4] <- paste0( dataDirectory, 'TrainingDataDog/' )

numberOfSubjectsPerCategory <- 1e6
for( i in 1:length( trainingDirectories ) )
  {
  trainingImageFilesPerCategory <- list.files( 
    path = trainingDirectories[i], pattern = "*.jpg", full.names = TRUE )
  numberOfSubjectsPerCategory <- min( numberOfSubjectsPerCategory,
    trainingProportion * length( trainingImageFilesPerCategory ) )
  }

trainingImageFiles <- c()
trainingClassifications <- c()
for( i in 1:length( trainingDirectories ) )
  {
  trainingImageFilesPerCategory <- list.files( 
    path = trainingDirectories[i], pattern = "*.jpg", full.names = TRUE )

  set.seed( 1234 )
  trainingIndices <- sample.int( 
    length( trainingImageFilesPerCategory ), size = numberOfSubjectsPerCategory )
  trainingImageFiles <- append( 
    trainingImageFiles, trainingImageFilesPerCategory[trainingIndices] )  
  trainingClassifications <- append( trainingClassifications, 
    rep( i-1, length( trainingIndices ) ) )
  }

trainingImages <- list()
trainingImageArrays <- list()
for ( i in 1:length( trainingImageFiles ) )
  {
  cat( "Reading ", trainingImageFiles[i], "\n" )
  trainingImages[[i]] <- readJPEG( trainingImageFiles[i] )
  if( length( dim( trainingImages[[i]] ) ) == 3 )
    {
    r <- as.matrix( resampleImage( 
          as.antsImage( trainingImages[[i]][,,1] ), 
          trainingImageSize, useVoxels = TRUE ) )
    r <- ( r - mean( r ) ) / sd( r )          
    g <- as.matrix( resampleImage( 
          as.antsImage( trainingImages[[i]][,,2] ), 
          trainingImageSize, useVoxels = TRUE ) )
    g <- ( g - mean( g ) ) / sd( g )      
    b <- as.matrix( resampleImage( 
          as.antsImage( trainingImages[[i]][,,3] ), 
          trainingImageSize, useVoxels = TRUE ) )
    b <- ( b - mean( b ) ) / sd( b )      
    } else {
    r <- as.matrix( resampleImage( 
          as.antsImage( trainingImages[[i]] ), 
          trainingImageSize, useVoxels = TRUE ) )
    r <- ( r - mean( r ) ) / sd( r )      
    g <- b <- r  
    }      
  trainingImageArrays[[i]] <- abind( r, g, b, along = 3 )  
  }

# trainingData <- abind( trainingImageArrays, along = 3 )  
# trainingData <- aperm( trainingData, c( 3, 1, 2 ) )
# X_train <- array( trainingData, dim = c( dim( trainingData ), 1 ) )

trainingData <- abind( trainingImageArrays, along = 4 )  
trainingData <- aperm( trainingData, c( 4, 1, 2, 3 ) )
X_train <- array( trainingData, dim = c( dim( trainingData ) ) )

segmentationLabels <- sort( unique( trainingClassifications ) )
numberOfLabels <- length( segmentationLabels )
Y_train <- to_categorical( trainingClassifications, numberOfLabels )

# resNetModel <- createResNetModel2D( c( dim( trainingImageArrays[[1]] ), 1 ),
#   numberOfClassificationLabels = numberOfLabels )
resNetModel <- createResNetModel2D( dim( trainingImageArrays[[1]] ),
  numberOfClassificationLabels = numberOfLabels, cardinality = 32 )

if( numberOfLabels == 2 )   
  {
  resNetModel %>% compile( loss = 'binary_crossentropy',
    optimizer = optimizer_adam( lr = 0.001 ),  
    metrics = c( 'binary_crossentropy', 'accuracy' ) )
  } else {
  resNetModel %>% compile( loss = 'categorical_crossentropy',
    optimizer = optimizer_adam( lr = 0.001 ),  
    metrics = c( 'categorical_crossentropy', 'accuracy' ) )
  }


track <- resNetModel %>% fit( X_train, Y_train, 
                 epochs = 40, batch_size = 32, verbose = 1, shuffle = TRUE,
                #  callbacks = list( 
                #    callback_model_checkpoint( paste0( baseDirectory, "resNetWeights.h5" ),
                #      monitor = 'val_loss', save_best_only = TRUE )
                  # callback_early_stopping( patience = 2, monitor = 'loss' ),
                  #  callback_reduce_lr_on_plateau( monitor = "val_loss", factor = 0.1 )
                #  ), 
                 validation_split = 0.2 )
# Save the model

# save_model_weights_hdf5( 
#   resNetModel, filepath = paste0( baseDirectory, 'resNetWeights.h5' ) )
# save_model_hdf5( 
#   resNetModel, filepath = paste0( baseDirectory, 'resNetModel.h5' ), overwrite = TRUE )

## Plot the model fitting

# epochs <- 1:length( track$metrics$loss )

# resNetModelDataFrame <- data.frame( Epoch = rep( epochs, 2 ), 
#                                   Type = c( rep( 'Training', length( epochs ) ), rep( 'Validation', length( epochs ) ) ),
#                                   Loss =c( track$metrics$loss, track$metrics$val_loss ), 
#                                   Accuracy = c( track$metrics$multilabel_dice_coefficient, track$metrics$val_multilabel_dice_coefficient )
#                                 )

# resNetModelLossPlot <- ggplot( data = resNetModelDataFrame, aes( x = Epoch, y = Loss, colour = Type ) ) +
#                  geom_point( shape = 1, size = 0.5 ) +
#                  geom_line( size = 0.3 ) +
#                  ggtitle( "Loss" )
                

# resNetModelAccuracyPlot <- ggplot( data = resNetModelDataFrame, aes( x = Epoch, y = Accuracy, colour = Type ) ) +
#                  geom_point( shape = 1, size = 0.5 ) +
#                  geom_line( size = 0.3 ) +
#                  ggtitle( "Accuracy")

# ggsave( paste0( baseDirectory, "resNetModelLossPlot.pdf" ), plot = resNetModelLossPlot, width = 5, height = 2, units = 'in' )
# ggsave( paste0( baseDirectory, "resNetModelAccuracyPlot.pdf" ), plot = resNetModelAccuracyPlot, width = 5, height = 2, units = 'in' )









