

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
denoised <- list()
#
for( i in 1:length( imageIDs ) )
  {
  cat( "Processing image", imageIDs[i], "\n" )
  img  = antsImageRead( getANTsRData( imageIDs[i] ) ) # %>% resampleImage( 2 )
  img = img + makeImage( img * 0 + 1, rnorm( prod( dim(img)),0,5)) # corrupt
  images[[i]] <- list( img )
  denoised[[i]] <- denoiseImage( img ) %>% denoiseImage() %>% n3BiasFieldCorrection( 4 )
  }
###############################################################
unetModel <- createUnetModel2D( c( dim( images[[1]][[1]] ), 1 ),
  numberOfFiltersAtBaseLayer = 16,
  numberOfLayers=6,
  dropoutRate = 0.0, strides=c(2,2),
  numberOfClassificationLabels = 1, mode='regression' )

# categorical_crossentropy
unetModel %>% compile( loss = 'mse',
  optimizer = optimizer_adam( ) )

sdAff = 0.08
mytd <- randomImageTransformBatchGenerator$new(
  imageList = images,
  outcomeImageList = denoised,
  transformType = "Affine",
  sdAffine = sdAff,
  imageDomain = images[[1]][[1]],
  toCategorical = FALSE )

tdgenfun <- mytd$generate( batchSize = 10 )
#
track <- unetModel$fit_generator(
  generator = reticulate::py_iterator( tdgenfun ),
  steps_per_epoch = 2,
  epochs = 400 )
################################################
################################################
k=1
mytd2 <- randomImageTransformBatchGenerator$new(
  imageList = images,
  outcomeImageList = denoised,
  transformType = "AffineAndDeformation",
  sdAffine = sdAff, spatialSmoothing = 8,
  imageDomain = images[[1]][[1]],
  toCategorical = FALSE )
tdgenfun2 <- mytd2$generate( batchSize = 10 )
testpop <- tdgenfun2()
domainMask = img * 0 + 1
testimg = makeImage( domainMask, testpop[[1]][k,,,1] )
plot( testimg )
predictedData <- unetModel %>% predict( testpop[[1]], verbose = 0 )
testimg = makeImage( domainMask, testpop[[1]][k,,,1] )
predimg = makeImage( domainMask, predictedData[k,,,1])
plot( predimg )
