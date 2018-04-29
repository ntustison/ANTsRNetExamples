library( ANTsR )
library( ANTsRNet )
library( keras )
library( jpeg )
library( reticulate )


keras::backend()$clear_session()
np <- reticulate::import( "numpy" )  

numberOfTrainingData <- 1000
inputImageSize <- c( 250, 250 )

visuallyInspectEachImage <- FALSE
baseDirectory <- './'
dataDirectory <- paste0( baseDirectory, './lfw_faces_tagged/' )
imageDirectory <- paste0( dataDirectory, 'Images/' )
annotationsDirectory <- paste0( dataDirectory, 'Annotations/' )
dataFile <- paste0( dataDirectory, 'data.csv' )
data <- read.csv( dataFile )  
uniqueImageFiles <- levels( as.factor( data$frame ) )

modelDirectory <- paste0( baseDirectory, '../../Models/' )

source( paste0( baseDirectory, 'ssdBatchGenerator.R' ) )

classes <- c( "eyes", "nose", "mouth" )
numberOfClassificationLabels <- length( classes ) + 1

###
#
# Read in the training image data. 
#
cat( "Reading images...\n" )
pb <- txtProgressBar( min = 0, max = numberOfTrainingData, style = 3 )

trainingImages <- list()
for( i in 1:numberOfTrainingData )
  {
  trainingImages[[i]] <- as.array( 
    readJPEG( paste0( imageDirectory, uniqueImageFiles[i] ) ) )

  if( i %% 100 == 0 )
    {
    gc( verbose = FALSE )
    }

  setTxtProgressBar( pb, i )
  }
cat( "\nDone.\n" )

###
#
# Read in the training boxes data. 
#

groundTruthLabels <- list()
for( i in 1:numberOfTrainingData )
  {
  groundTruthBoxes <- data[which( data$frame == uniqueImageFiles[i] ),]
  image <- trainingImages[[i]]
  groundTruthBoxes <- 
    data.frame( groundTruthBoxes[, 6], groundTruthBoxes[, 2:5]  )
  colnames( groundTruthBoxes ) <- c( "class_id", 'xmin', 'xmax', 'ymin', 'ymax' )

  groundTruthLabels[[i]] <- groundTruthBoxes

  if( visuallyInspectEachImage == TRUE )
    {
    cat( "Drawing", uniqueImageFiles[i], "\n" )

    classIds <- groundTruthBoxes[, 1]

    boxColors <- c()
    boxCaptions <- c()
    for( j in 1:length( classIds ) )
      {
      boxColors[j] <- rainbow( 
        length( classes ) )[which( classes[classIds[j]] == classes )]
      boxCaptions[j] <- classes[which( classes[classIds[j]] == classes )]
      }  
    drawRectangles( image, groundTruthBoxes[, 2:5], boxColors = boxColors, 
      captions = boxCaptions )
    readline( prompt = "Press [enter] to continue " )
    }  
  }  

if( visuallyInspectEachImage == TRUE )
  {
  cat( "\n\nDone inspecting images.\n" )
  }

###
#
# Create the SSD model
#

ssdOutput <- createSsd7Model2D( c( inputImageSize, 3 ), 
  numberOfClassificationLabels = numberOfClassificationLabels
  )

ssdModel <- ssdOutput$ssdModel 
anchorBoxes <- ssdOutput$anchorBoxes

load_model_weights_hdf5( ssdModel, 
  filepath = paste0( baseDirectory, 'ssd7Weights.h5' ) )

optimizerAdam <- optimizer_adam( 
  lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = 5e-05 )

ssdLoss <- LossSSD$new( dimension = 2L, backgroundRatio = 3L, 
  minNumberOfBackgroundBoxes = 0L, 
  alpha = 1.0, numberOfClassificationLabels = numberOfClassificationLabels )

ssdModel %>% compile( loss = ssdLoss$compute_loss, optimizer = optimizerAdam )

yaml_string <- model_to_yaml( ssdModel )
writeLines( yaml_string, paste0( baseDirectory, "ssd7Model.yaml" ) )
json_string <- model_to_json( ssdModel )
writeLines( json_string, paste0( baseDirectory, "ssd7Model.json" ) )

###
#
#  Debugging:  draw all anchor boxes
#

# image <- readJPEG( paste0( imageDirectory, uniqueImageFiles[1] ) )
# for( i in 1:length( anchorBoxes) )
#   {
#   cat( "Drawing anchor box:", i, "\n" )
#   anchorBox <- anchorBoxes[[i]]
#   anchorBox[, 1:2] <- anchorBox[, 1:2]
#   anchorBox[, 3:4] <- anchorBox[, 3:4]
#   drawRectangles( image, anchorBox[,], 
#     boxColors = rainbow( nrow( anchorBox[,] ) ) )
#   readline( prompt = "Press [enter] to continue\n" )
#   # for( j in 1:nrow( anchorBoxes[[i]] ) )
#   #   {
#   #   cat( "Drawing anchor box:", i, ",", j, "\n" )
#   #   anchorBox <- anchorBoxes[[i]][j,]
#   #   anchorBox[1:2] <- anchorBox[1:2] * ( inputImageSize[1] - 2 ) + 1
#   #   anchorBox[3:4] <- anchorBox[3:4] * ( inputImageSize[2] - 2 ) + 1
#   #   drawRectangles( image, anchorBox, boxColors = "red" )
#   #   readline( prompt = "Press [enter] to continue\n" )
#   #   }
#   }

###
#
#  Debugging:  visualize corresponding anchorBoxes
#

if( visuallyInspectEachImage == TRUE )
  {
  Y_train <- encodeSsd2D( groundTruthLabels, anchorBoxes, inputImageSize, rep( 1.0, 4 ) )

  for( i in 1:numberOfTrainingData )
    {
    cat( "Drawing", i, "\n" )
    image <- trainingImages[[i]]

    # Get anchor boxes  
    singleY <- Y_train[i,,]

    if( length( classes ) > 1 )
      {
      singleY <- singleY[which( rowSums( 
        singleY[, 2:numberOfClassificationLabels] ) > 0 ),]
      } else {
      singleY <- singleY[which( singleY[, 2] > 0 ),]
      }

    if( !is.null( dim( singleY ) ) )
      {     
      anchorClassIds <- max.col( singleY[, 2:numberOfClassificationLabels] )
      } else {
      anchorClassIds <- which.max( singleY[2:numberOfClassificationLabels] )
      singleY <- matrix( singleY, nrow = 1 )
      }

    anchorBoxColors <- c()
    anchorBoxCaptions <- c()
    for( j in 1:length( anchorClassIds ) )
      {
      anchorBoxColors[j] <- rainbow( 
        length( classes ) )[which( classes[anchorClassIds[j]] == classes )]
      anchorBoxCaptions[j] <- classes[which( classes[anchorClassIds[j]] == classes )]
      }

    # Get truth boxes
    truthLabel <- groundTruthLabels[[i]]

    truthClassIds <- truthLabel[, 1]
    truthColors <- c()
    truthCaptions <- c()
    for( j in 1:length( truthClassIds ) )
      {
      truthColors[j] <- rainbow( 
        length( classes ) )[which( classes[truthClassIds[j]] == classes )]
      truthCaptions[j] <- classes[which( classes[truthClassIds[j]] == classes )]
      }

    # Convert from offsets to absolute coordinates

    indices <- numberOfClassificationLabels + -1:4
    singleYConverted <- np$copy( singleY[, indices, drop = FALSE] )
    singleYConverted[, 1] <- anchorClassIds
    singleYConverted[, 2] <- 0.25   # confidence values

    indices1 <- numberOfClassificationLabels + 11:12
    indices2 <- numberOfClassificationLabels + 7:8
    indices3 <- numberOfClassificationLabels + 9:10
    indices4 <- numberOfClassificationLabels + 5:6

    # singleYConverted[, c( 5, 6 )] <- np$exp( singleYConverted[, c( 5, 6 )] * singleY[, indices1] ) 
    singleYConverted[, c( 5, 6 )] <- np$exp( 0 * singleY[, indices1] ) 
    singleYConverted[, c( 5, 6 )] <- singleYConverted[, c( 5, 6 )] * singleY[, indices2] 
    singleYConverted[, c( 3, 4 )] <- singleYConverted[, c( 3, 4 )] * ( singleY[, indices3] * singleY[, indices2] ) 
    singleYConverted[, c( 3, 4 )] <- singleYConverted[, c( 3, 4 )] + singleY[, indices4] 

    singleYConverted[,3:6] <- convertCoordinates( singleYConverted[,3:6], type = 'centroids2minmax' )

    if( !is.null( dim( singleY ) ) )
      {
      boxes <- rbind( singleYConverted[, 3:6], as.matrix( truthLabel[, 2:5] ) )
      } else {
      boxes <- rbind( singleYConverted[, 3:6], as.matrix( truthLabel[, 2:5] ) )
      }
    boxColors <- c( anchorBoxColors, truthColors )
    confidenceValues <- c( singleYConverted[,2], rep( 1.0, length( truthColors ) ) )

    drawRectangles( image, boxes, boxColors = boxColors, confidenceValues = confidenceValues )

    readline( prompt = "Press [enter] to continue " )
    }
  }  

###
#
# Set up the training generator
#
batchSize <- 50L

# Split trainingData into "training" and "validation" componets for
# training the model.
sampleIndices <- sample( numberOfTrainingData )

validationSplit <- 900 # round( ( 1 - 0.2 ) * numberOfTrainingData )
trainingIndices <- sampleIndices[1:validationSplit]
validationIndices <- sampleIndices[( validationSplit + 1 ):numberOfTrainingData]

trainingData <- ssdImageBatchGenerator$new( 
  imageList = trainingImages[trainingIndices], 
  labels = groundTruthLabels[trainingIndices] )

# trainingDataGenerator <- trainingData$generate( batchSize = batchSize,
#   anchorBoxes = anchorBoxes, variances = rep( 1.0, 4 ), equalize = NULL,
#   brightness = NULL, flipHorizontally = NULL, translate = NULL, 
#   scale = NULL )

trainingDataGenerator <- trainingData$generate( batchSize = batchSize,
  anchorBoxes = anchorBoxes, variances = rep( 1.0, 4 ), equalize = NULL,
  brightness = c( 0.5, 2, 0.5 ), flipHorizontally = 0.5, 
  translate = list( c( 5, 50 ), c( 3, 30 ), 0.5 ), 
  scale = c( 0.75, 1.3, 0.5 ) )

validationData <- ssdImageBatchGenerator$new( 
  imageList = trainingImages[validationIndices], 
  labels = groundTruthLabels[validationIndices] )

validationDataGenerator <- validationData$generate( batchSize = batchSize,
  anchorBoxes = anchorBoxes, variances = rep( 1.0, 4 ), equalize = NULL,
  brightness = NULL, flipHorizontally = NULL, 
  translate = NULL, 
  scale = NULL )

###
#
# Run training
#

track <- ssdModel$fit_generator( 
  generator = reticulate::py_iterator( trainingDataGenerator ), 
  steps_per_epoch = 20, #ceiling( length( trainingIndices ) / batchSize ),
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
