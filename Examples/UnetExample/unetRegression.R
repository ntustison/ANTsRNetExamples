library( ANTsRNet )
library( ANTsR )
library( abind )
library( keras )
##########################################
imageIDs <- c( "r16", "r27", "r30",  "r62", 'r64' )
imageIDs2 <- c( "r85", "r27" ) # testing
hiSpc = c( 1, 1 )
loSpc = c( 4, 4 )
################################################
images <- list()
denoised <- list()
for( j in 1:50 )
  {
  i = j %% 6
  if ( i == 0 ) i = 1
  cat( "Processing image", imageIDs[i], "\n" )
  img = antsImageRead( getANTsRData( imageIDs[i] ) ) %>% iMath("Normalize")
  loimg = resampleImage( img, loSpc ) %>% resampleImageToTarget( img )
  images[[i]] <- list( loimg, iMath( loimg, "Laplacian" ),
    iMath( loimg, "Grad" )  )
  denoised[[i]] <- img
  }
################################################ testing data #
images2 <- list()
denoised2 <- list()
for( i in 1:length( imageIDs2 ) )
  {
  cat( "Processing image", imageIDs2[i], "\n" )
  img = antsImageRead( getANTsRData( imageIDs2[i] ) ) %>% iMath("Normalize")
  loimg = resampleImage( img, loSpc ) %>% resampleImageToTarget( img )
  images2[[i]] <- list( loimg, iMath( loimg, "Laplacian" ),
    iMath( loimg, "Grad" )  )
  denoised2[[i]] <- img
  }
###############################################################
nChannels = length( images[[i]] )
unetModel <- createUnetModel2D(
    c( dim( images[[1]][[1]] ), nChannels ),
    numberOfFiltersAtBaseLayer = 16,
    numberOfLayers = 6,
    dropoutRate = 0.0, strides=c(2,2),
    numberOfOutputs = 1, mode='regression' )

# custom metric via ntustison
unetModel %>% compile( loss = 'mse',
  optimizer = optimizer_adam( lr = 0.001 ),
  metrics = c( 'mse' ) )

# affine augmentation
sdAff = 0.03
mytd <- randomImageTransformBatchGenerator$new(
  imageList = images,
  outcomeImageList = denoised,
  transformType = "Affine",
  sdAffine = sdAff,
  imageDomain = images[[1]][[1]],
  imageDomainY = loimg,
  toCategorical = FALSE )
tdgenfun <- mytd$generate( batchSize = 2 )
################################################
unetModel %>% fit_generator(
  generator = tdgenfun,
  steps_per_epoch = 1,
  epochs = 5 )
################################################
mytd2 <- randomImageTransformBatchGenerator$new(
  imageList = images2,
  outcomeImageList = denoised2,
  transformType = "Affine",
  sdAffine = sdAff,
  imageDomain = images[[1]][[1]],
  imageDomainY = loimg,
  toCategorical = FALSE )
tdgenfun2 <- mytd2$generate( batchSize = 8 )
testpop <- tdgenfun2()
domainMaskX = loimg * 0 + 1
domainMaskY = img * 0 + 1
predictedData <- unetModel %>% predict( testpop[[1]], verbose = 0 )
testimgC1 = makeImage( domainMaskX, testpop[[1]][1,,,1] )
predimg = makeImage( domainMaskY, predictedData[1,,,1])
# show side by side results on test data
layout( matrix(c(1,2), 1, 2, byrow = TRUE))
plot( iMath(testimgC1, "Normalize"), doCropping = F )
plot( iMath(predimg, "Normalize"), doCropping = F )
