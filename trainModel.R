library( ANTsR )
library( keras )
library( abind )

baseDirectory <- '/Users/ntustison/Data/UNet/'
dataDirectory <- paste0( baseDirectory, 'Images/' )
trainingDirectory <- paste0( dataDirectory, 'TrainingData/' )
testingDirectory <- paste0( dataDirectory, 'TestingData/' )

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
  }

trainingData <- abind( trainingImageArrays, along = 3 )  
trainingData <- aperm( trainingData, c( 3, 1, 2 ) )

trainingLabelData <- abind( trainingMaskArrays, along = 3 )  
trainingLabelData <- aperm( trainingLabelData, c( 3, 1, 2 ) )

numberOfLabels <- 3 

X_train <- array( trainingData, dim = c( dim( trainingData ), 1 ) )
Y_train <- array( to_categorical( trainingLabelData ), dim = c( dim( trainingData ), numberOfLabels ) )

unetModel <- createUnetModel2D( dim( trainingImageArrays[[1]] ), numberOfLabels )
track <- unetModel %>% fit( X_train, Y_train,
                 epochs = 150, batch_size = 10,
                 callbacks = callback_early_stopping(patience = 2, monitor = 'acc'),
                 validation_split = 0.3 )

