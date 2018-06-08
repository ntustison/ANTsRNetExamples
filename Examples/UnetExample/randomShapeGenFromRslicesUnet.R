

# Sys.setenv(TENSORFLOW_PYTHON='/usr/local/bin/python3')
library( ANTsRNet )
library( ANTsR )
library(abind)
library( keras )
imageIDs <- c( "r16", "r27", "r30", "r62", "r64", "r85" )
# Perform simple 3-tissue segmentation.  For convenience we are going
segmentationLabels <- c( 1, 2, 3 )
numberOfLabels <- length( segmentationLabels )

images <- list()
kmeansSegs <- list()
#
for( i in 1:length( imageIDs ) )
  {
  cat( "Processing image", imageIDs[i], "\n" )
  img  = antsImageRead( getANTsRData( imageIDs[i] ) ) %>% resampleImage( 2 )
  images[[i]] <- list( img )
  kmeansSegs[[i]] <- thresholdImage( img, "Otsu", 3 )
  }
###
unetModel <- createUnetModel2D( c( dim( images[[1]][[1]] ), 1 ),
  numberOfFiltersAtBaseLayer = 32, dropoutRate = 0.1,
  numberOfClassificationLabels = numberOfLabels+1 )

# categorical_crossentropy
unetModel %>% compile( loss = 'categorical_crossentropy',
  optimizer = optimizer_adam( ),
  metrics = c( multilabel_dice_coefficient ) )

mytd <- randomImageTransformBatchGenerator$new(
  imageList = images,
  outcomeImageList = kmeansSegs,
  transformType = "Affine",
  sdAffine = 0.05,
  imageDomain = images[[1]][[1]],
  toCategorical = TRUE )

tdgenfun <- mytd$generate( batchSize = 16  )
#
track <- unetModel$fit_generator(
  generator = reticulate::py_iterator( tdgenfun ),
  steps_per_epoch = 1,
  epochs = 999  )


testpop <- tdgenfun()


predictedData <- unetModel %>% predict( testpop[[1]], verbose = 0 )
probabilityImagesArray <- decodeUnet( predictedData, img )
k=1
for ( t in 1:4 ) {
  plot( makeImage( img * 0 + 1, testpop[[1]][k,,,1] ), color.overlay='red',
    probabilityImagesArray[[k]][[t]], alpha=0.8, window.overlay=c(0.5,1) )
    Sys.sleep(1)
  }

# this demonstrates the augmentation style visually
if ( FALSE ) {
  library( ANTsRNet )
  library( ANTsR )
  imageIDs <- c( "r16", "r27", "r30", "r62", "r64", "r85" )
  kmeansSegsDemo = list()
  imagesDemo = list()
  for( i in 1:length( imageIDs ) )
    {
    cat( "Processing image", imageIDs[i], "\n" )
    imgDemo  = antsImageRead( getANTsRData( imageIDs[i] ) )
    imagesDemo[[i]] <- list( imgDemo )
    kmeansSegsDemo[[i]] <- thresholdImage( imgDemo, "Otsu", 3 )
    }
  rand = randomImageTransformAugmentation( imgDemo, imagesDemo,
    kmeansSegsDemo, n = 32, typeOfTransform = 'Affine', sdAffine = 0.2 )
  for ( jj in 1:length( rand$outputPredictorList ) ) {
    plot( rand$outputPredictorList[[jj]][[1]],
      rand$outputOutcomeList[[jj]], alpha=0.1, doCropping=FALSE )
    Sys.sleep( 3 )
  }
  # reduce / increase the value of sdAffine to create less/more variation
}
