library( ANTsRNet )
library( ANTsR )
library( abind )
library( keras )
# for plaidml
# use_implementation(implementation = c("keras"))
# use_backend(backend = 'plaidml' )
##########################################
imageIDs <- c( "r16", "r27", "r30",  "r62", "r85" )

images <- list()
hiRes <- list()
hiSpc = c( 1, 1 )
loSpc = c( 2, 2 )
for( i in 1:length( imageIDs ) )
  {
  cat( "Processing image", imageIDs[i], "\n" )
  img = antsImageRead( getANTsRData( imageIDs[i] ) ) %>% iMath("Normalize")
  loimg = resampleImage( img, loSpc )
  images[[i]] <- list( loimg )
  hiRes[[i]] <- img
  }

###############################################################
nChannels = length( images[[i]] )

srModel <- createResNetSuperResolutionModel2D(
  c( dim( images[[1]][[1]] ), nChannels ),
  convolutionKernelSize = c(3, 3),
  numberOfFilters = 32,
  numberOfResidualBlocks = 2 )

srModel %>% compile( loss = loss_peak_signal_to_noise_ratio_error,
  optimizer = optimizer_adam( lr = 0.001 ),
  metrics = c( 'mse', peak_signal_to_noise_ratio ) )

sdAff = 0.03
mytd <- randomImageTransformBatchGenerator$new(
  imageList = images,
  outcomeImageList = hiRes,
  transformType = "Affine",
  sdAffine = sdAff,
  imageDomain = images[[1]][[1]],
  imageDomainY = hiRes[[1]],
  toCategorical = FALSE )

tdgenfun <- mytd$generate( batchSize = 8 )

track <- srModel$fit_generator(
  generator = reticulate::py_iterator( tdgenfun ),
  steps_per_epoch = 5,
  epochs = 13 )
################################################
################################################
testpop <- tdgenfun()
domainMaskX = loimg * 0 + 1
domainMaskY = img * 0 + 1
predictedData <- srModel %>% predict( testpop[[1]], verbose = 0 )
testimgC1 = makeImage( domainMaskX, testpop[[1]][1,,,1] )
predimg = makeImage( domainMaskY, predictedData[1,,,1])
