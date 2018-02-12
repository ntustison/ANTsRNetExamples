library( ANTsR )
library( xml2 )
library( tidyverse )
library( stringr )
library( keras )
library( ggplot2 )
library( jpeg )

parseXML <- function( xml, labels ) {
  
  frame <- xml %>%
    xml_find_first("//filename") %>%
    xml_text()
  
  classes <- xml %>%
    xml_find_all("//object") %>%
    xml_find_all(".//name") %>%
    xml_text() %>%
    factor( levels = labels ) %>%
    as.integer() %>%
    as_tibble() %>%
    magrittr::set_colnames( "class_id" )
  
  bndbx <- xml %>%
    xml_find_all("//bndbox") %>%
    xml_children() %>%
    xml_integer() %>%
    split( rep( 1:dim( classes )[1], each = 4 ) ) %>%
    as_tibble() %>%
    t() %>%
    magrittr::set_colnames(c("xmin", "ymin", "xmax", "ymax")) %>%
    as_tibble() %>%
    select(xmin, xmax, ymin, ymax)
  
  cbind(frame, bndbx, classes) %>%
    as_tibble %>%
    mutate(frame = as.character(frame))
  }

baseDirectory <- './'
dataDirectory <- paste0( baseDirectory, './lfw_faces_tagged/' )
imageDirectory <- paste0( dataDirectory, 'Images/' )
annotationsDirectory <- paste0( dataDirectory, 'Annotations/' )
dataFile <- paste0( dataDirectory, 'data.csv' )

modelDirectory <- paste0( baseDirectory, '../../Models/' )

classes <- c( "eyes", "nose", "mouth" )

if( ! file.exists( dataFile ) )
  {
  data <- list.files( annotationsDirectory, full.names = TRUE ) %>%
    discard( !str_detect( ., "xml" ) ) %>%
    map( ., read_xml ) %>%
    map_dfr( parseXML, classes )

  data <- list.files( annotationsDirectory, full.names = TRUE ) %>%
    discard( !str_detect( ., "xml" ) ) %>%
    map( ., read_xml ) %>%
    map_dfr( parseXML, classes )
  data <- data[complete.cases( data ),]

  write.csv( data, dataFile, row.names = FALSE )
  } else {
  data <- read.csv( dataFile )  
  }

###
#
# Read in the training data.  There are 1000 total images.  Read in 800
# for training and then read the remaining data for testing/prediction.
#

numberOfTrainingData <- 800
trainingImageFiles <- rep( NA, numberOfTrainingData )
for( i in 1:numberOfTrainingData )
  {
  trainingImageFiles[i] <- paste0( imageDirectory, data$frame[i] )  
  }

inputImageSize <- c( 300, 300 )
trainingData <- array( dim = c( numberOfTrainingData, inputImageSize, 3 ) )

cat( "Reading images...\n" )
pb <- txtProgressBar( min = 0, max = numberOfTrainingData, style = 3 )
for ( i in 1:length( trainingImageFiles ) )
  {
  # cat( "Reading ", trainingImageFiles[i], "\n" )
  trainingImage <- readJPEG( trainingImageFiles[i] )

  r <- as.matrix( resampleImage( 
        as.antsImage( trainingImage[,,1] ), 
        inputImageSize, useVoxels = TRUE ) )
  r <- ( r - mean( r ) ) / sd( r )          
  g <- as.matrix( resampleImage( 
        as.antsImage( trainingImage[,,2] ), 
        inputImageSize, useVoxels = TRUE ) )
  g <- ( g - mean( g ) ) / sd( g )      
  b <- as.matrix( resampleImage( 
        as.antsImage( trainingImage[,,3] ), 
        inputImageSize, useVoxels = TRUE ) )
  b <- ( b - mean( b ) ) / sd( b )      

  trainingData[i,,,1] <- r 
  trainingData[i,,,2] <- g 
  trainingData[i,,,3] <- b 

  if( i %% 100 == 0 )
    {
    gc( verbose = FALSE )
    }

  setTxtProgressBar( pb, i )
  }
cat( "\nDone.\n" )

X_train <- trainingData

###
#
# Create the SSD model
#

source( paste0( modelDirectory, 'createSsdModel.R' ) )

# Input size must be greater than >= 258 for a single dimension

inputImageSize <- c( inputImageSize, 3 )
ssdOutput <- createSsdModel2D( inputImageSize, 
  numberOfClassificationLabels = length( classes ) + 1,
  )

ssdModel <- ssdOutput$ssdModel 
anchorBoxes <- ssdOutput$anchorBoxes

###
#
# Create the Y encoding
#
uniqueImageFiles <- levels( as.factor( data$frame ) )

groundTruthLabels <- list()
for( i in 1:numberOfTrainingData )
  {
  groundTruthBoxes <- data[which( data$frame == uniqueImageFiles[i] ),]
  groundTruthBoxes <- 
    data.frame( groundTruthBoxes[, 6], groundTruthBoxes[, 2:5] )
  colnames( groundTruthBoxes ) <- c( "class_id", 'xmin', 'xmax', 'ymin', 'ymax' )
  groundTruthLabels[[i]] <- groundTruthBoxes
  }

Y_train <- encodeY( groundTruthLabels, anchorBoxes, rep( 1.0, 4 ) )

optimizerAdam <- optimizer_adam( 
  lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = 5e-05 )

ssdLoss <- lossSsd$new( backgroundRatio = 3L, minNumberOfBackgroundBoxes = 0L, 
  alpha = 1.0, numberOfClassificationLabels = length( classes ) + 1 )

ssdModel %>% compile( loss = ssdLoss$compute_loss, optimizer = optimizerAdam )

track <- ssdModel %>% fit( X_train, Y_train, 
                 epochs = 40, batch_size = 32, verbose = 1, shuffle = TRUE,
                 callbacks = list( 
                   callback_model_checkpoint( paste0( baseDirectory, "ssdWeights.h5" ), 
                     monitor = 'val_loss', save_best_only = TRUE )
                  # callback_early_stopping( patience = 2, monitor = 'loss' ),
                  #  callback_reduce_lr_on_plateau( monitor = "val_loss", factor = 0.1 )
                 ), 
                 validation_split = 0.2 )
