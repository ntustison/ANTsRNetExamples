library( ANTsR )
library( ANTsRNet
library( xml2 )
library( tidyverse )
library( stringr )
library( keras )
library( ggplot2 )
library( jpeg )

warning( paste0( 
  "Note that the original architecture was initialized with the ",
  "pre-trained VGG-16 weights which is not done here yet.  As mentioned here ",
  "https://github.com/pierluigiferrari/ssd_keras \n",
  "\"It is strongly recommended that you load the pre-trained VGG-16 ",
  "weights when attempting to train an SSD300 or SSD512, otherwise ",
  "your training will almost certainly be unsuccessful. Note that ",
  "the original VGG-16 was trained layer-wise, so trying to train ",
  "the even deeper SSD300 all at once from scratch will very likely ",
  "fail. Also note that even with the pre-trained VGG-16 weights it ",
  "will take at least ~20,000 training steps to get a half-decent ",
  "performance out of SSD300.\"\n"
  ) )

keras::backend()$clear_session()

numberOfTrainingData <- 800

visuallyInspectEachImage <- FALSE

baseDirectory <- './'
dataDirectory <- paste0( baseDirectory, './lfw_faces_tagged/' )
imageDirectory <- paste0( dataDirectory, 'Images/' )
annotationsDirectory <- paste0( dataDirectory, 'Annotations/' )
dataFile <- paste0( dataDirectory, 'data.csv' )

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
uniqueImageFiles <- levels( as.factor( data$frame ) )

###
#
# Read in the training data.  There are 1000 total images.  Read in 800
# for training and then read the remaining data for testing/prediction.
#

trainingImageFiles <- rep( NA, numberOfTrainingData )
for( i in 1:numberOfTrainingData )
  {
  trainingImageFiles[i] <- paste0( imageDirectory, uniqueImageFiles[i] )  
  }

# original images are 250 x 250 so we need to multiply the points by 
# 300 / 250 = 1.2
scaleFactor <- 1.2

inputImageSize <- c( 300, 300 )
trainingData <- array( dim = c( 2 * numberOfTrainingData, inputImageSize, 3 ) )

cat( "Reading images...\n" )
pb <- txtProgressBar( min = 0, max = numberOfTrainingData, style = 3 )
for ( i in 1:length( trainingImageFiles ) )
  {
  # cat( "Reading ", trainingImageFiles[i], "\n" )
  trainingImage <- readJPEG( trainingImageFiles[i] )

  r <- as.matrix( resampleImage( 
        as.antsImage( trainingImage[,,1] ), 
        inputImageSize, useVoxels = TRUE ) )
  g <- as.matrix( resampleImage( 
        as.antsImage( trainingImage[,,2] ), 
        inputImageSize, useVoxels = TRUE ) )
  b <- as.matrix( resampleImage( 
        as.antsImage( trainingImage[,,3] ), 
        inputImageSize, useVoxels = TRUE ) )
  
  r <- ( r - min( r ) ) / ( max( r ) - min( r ) )
  g <- ( g - min( g ) ) / ( max( g ) - min( g ) )
  b <- ( b - min( b ) ) / ( max( b ) - min( b ) )

  trainingData[2*i-1,,,1] <- r
  trainingData[2*i-1,,,2] <- g
  trainingData[2*i-1,,,3] <- b

  # Flip the images horizontally

  flipIndices <- seq( from = inputImageSize[1], to = 1, by = -1 )

  trainingData[2*i,,,1] <- r[,flipIndices]
  trainingData[2*i,,,2] <- g[,flipIndices]
  trainingData[2*i,,,3] <- b[,flipIndices]
  
  if( i %% 100 == 0 )
    {
    gc( verbose = FALSE )
    }

  setTxtProgressBar( pb, i )
  }
cat( "\nDone.\n" )

X_train <- trainingData

# Input size must be greater than >= 258 for a single dimension

ssdOutput <- createSsdModel2D( c( inputImageSize, 3 ), 
  numberOfClassificationLabels = length( classes ) + 1,
  )

ssdModel <- ssdOutput$ssdModel 
anchorBoxes <- ssdOutput$anchorBoxes

yaml_string <- model_to_yaml( ssdModel )
writeLines( yaml_string, paste0( baseDirectory, "ssd300Model.yaml" ) )
json_string <- model_to_json( ssdModel )
writeLines( json_string, paste0( baseDirectory, "ssd300Model.json" ) )

###
#
# Create the Y encoding
#
uniqueImageFiles <- levels( as.factor( data$frame ) )

groundTruthLabels <- list()
for( i in 1:numberOfTrainingData )
  {
  for( j in 0:1 )  
    {
    groundTruthBoxes <- data[which( data$frame == uniqueImageFiles[i] ),]
    image <- trainingData[2*i-1+j,,,]
    groundTruthBoxes <- 
      data.frame( groundTruthBoxes[, 6], groundTruthBoxes[, 2:5] * scaleFactor  )
    colnames( groundTruthBoxes ) <- c( "class_id", 'xmin', 'xmax', 'ymin', 'ymax' )
    if( j == 1 )
      {
      groundTruthBoxes[, 2:3] <- inputImageSize[1] - groundTruthBoxes[, 2:3] 
      groundTruthBoxes[, 2:3] <- groundTruthBoxes[, seq( 3, 2, by = -1 )]
      }
    groundTruthLabels[[2*i-1+j]] <- groundTruthBoxes

    if( visuallyInspectEachImage == TRUE )
      {
      cat( "Drawing", trainingImageFiles[i], "\n" )

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
  }  

if( visuallyInspectEachImage == TRUE )
  {
  cat( "\n\nDone inspecting images.\n" )
  }

Y_train <- encodeSsd2D( groundTruthLabels, anchorBoxes, inputImageSize, rep( 1.0, 4 ) )

###
#
#  Debugging:  draw all anchor boxes
#

# image <- readJPEG( trainingImageFiles[1] )
# for( i in 1:length( anchorBoxes) )
#   {
#   # cat( "Drawing anchor box:", i, "\n" )
#   # anchorBox <- anchorBoxes[[i]]
#   # anchorBox[, 1:2] <- anchorBox[, 1:2] * ( inputImageSize[1] - 2 ) + 1
#   # anchorBox[, 3:4] <- anchorBox[, 3:4] * ( inputImageSize[2] - 2 ) + 1
#   # drawRectangles( image, anchorBox[,], 
#   #   boxColors = rainbow( nrow( anchorBox[,] ) ) )
#   # readline( prompt = "Press [enter] to continue\n" )
#   for( j in 1:nrow( anchorBoxes[[i]] ) )
#     {
#     cat( "Drawing anchor box:", i, ",", j, "\n" )
#     anchorBox <- anchorBoxes[[i]][j,]
#     anchorBox[1:2] <- anchorBox[1:2] * ( inputImageSize[1] - 2 ) + 1
#     anchorBox[3:4] <- anchorBox[3:4] * ( inputImageSize[2] - 2 ) + 1
#     drawRectangles( image, anchorBox, boxColors = "red" )
#     readline( prompt = "Press [enter] to continue\n" )
#     }
#   }

###
#
#  Debugging:  visualize corresponding anchorBoxes
#

if( visuallyInspectEachImage == TRUE )
  {
  for( i in 1:numberOfTrainingData )
    {
    cat( "Drawing", trainingImageFiles[i], "\n" )
    image <- trainingData[i,,,]

    # Get anchor boxes  
    singleY <- Y_train[i,,]
    singleY <- singleY[which( rowSums( 
      singleY[, 2:( 1 + length( classes ) )] ) > 0 ),]

    numberOfClassificationLabels <- length( classes ) + 1
    xIndices <- numberOfClassificationLabels + 5:6
    singleY[, xIndices] <- singleY[, xIndices] * ( inputImageSize[1] - 2 ) + 1
    yIndices <- numberOfClassificationLabels + 7:8
    singleY[, yIndices] <- singleY[, yIndices] * ( inputImageSize[2] - 2 ) + 1

    anchorClassIds <- max.col( singleY[, 1:4] ) - 1

    anchorBoxColors <- c()
    anchorBoxCaptions <- c()
    for( j in 1:length( anchorClassIds ) )
      {
      anchorBoxColors[j] <- rainbow( 
        length( classes ) )[which( classes[anchorClassIds[j]] == classes )]
      # anchorBoxCaptions[j] <- classes[which( classes[anchorClassIds[j]] == classes )]
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

    boxes <- rbind( singleY[, 9:12], as.matrix( truthLabel[, 2:5] ) )
    boxColors <- c( anchorBoxColors, truthColors )
    confidenceValues <- c( rep( 0.2, length( anchorBoxColors ) ), rep( 1.0, length( truthColors ) ) )

    drawRectangles( image, boxes, boxColors = boxColors, confidenceValues = confidenceValues )

    readline( prompt = "Press [enter] to continue " )
    }
  }  

optimizerAdam <- optimizer_adam( 
  lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = 5e-04 )

ssdLoss <- LossSSD$new( backgroundRatio = 3L, minNumberOfBackgroundBoxes = 0L, 
  alpha = 1.0, numberOfClassificationLabels = length( classes ) + 1 )

ssdModel %>% compile( loss = ssdLoss$compute_loss, optimizer = optimizerAdam )

track <- ssdModel %>% fit( X_train, Y_train, 
                 epochs = 40, batch_size = 32, verbose = 1, shuffle = TRUE,
                 callbacks = list( 
                   callback_model_checkpoint( paste0( baseDirectory, "ssd300Weights.h5" ), 
                     monitor = 'val_loss', save_best_only = TRUE )
                  # callback_early_stopping( patience = 2, monitor = 'loss' ),
                  #  callback_reduce_lr_on_plateau( monitor = "val_loss", factor = 0.1 )
                 ), 
                 validation_split = 0.2 )


