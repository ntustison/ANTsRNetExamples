

library( ANTsRNet )
library( ANTsR )
library( abind )
library( keras )
imageIDs <- c( "r16", "r27", "r30", "r62", "r64", "r85" )
images <- list()
scl = 4
leaveout = 4
sdt = 0.1
if ( ! exists( "myep" ) ) myep = 500 # reasonable default
ref = ri( 16 )
for( i in 1:length( imageIDs ) )
  {
  cat( "Processing image", imageIDs[i], "\n" )
  img  = antsImageRead( getANTsRData( imageIDs[i] ) )
  reg = antsRegistration( ref, img, "Affine" )
  images[[i]] <- ( iMath( reg$warpedmovout, "Normalize" ) * 255 ) %>%
    resampleImage( scl )
  }
ref = ri( 16 )  %>% resampleImage( scl )

build_model <- function( input_shape, num_regressors ) {

  # Define model
  myact='relu'
#  myact='linear'
  model <- keras_model_sequential() %>%
    layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = myact,
                  input_shape = input_shape) %>%
    layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = myact) %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_dropout( rate = 0.0 ) %>%
    layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = myact) %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_dropout( rate = 0.0 ) %>%
    layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = myact) %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_dropout( rate = 0.0 ) %>%
    layer_flatten() %>%
    layer_dense(units = 32, activation = myact) %>%
    layer_dropout(rate = 0.0 ) %>%
    layer_dense(units = num_regressors )

  model %>% compile(
#    loss = "cosine_proximity",
    loss = "mean_absolute_error",
#    optimizer = optimizer_rmsprop(),
    optimizer = optimizer_adam(  amsgrad = TRUE ),
#    optimizer = optimizer_sgd( ),
    metrics = list("mean_absolute_error")
  )

  model
}

affTx = createAntsrTransform( "AffineTransform", dimension = 2 )
numRegressors = length( getAntsrTransformParameters( affTx ) )
input_shape <- c( dim( images[[1]]), 1)

regressionModel <- build_model(  input_shape, numRegressors   )
regressionModel %>% summary()


mytd <- randomImageTransformParametersBatchGenerator$new(
  imageList = images[ -leaveout ],
  transformType = "Affine",
  sdTransform = sdt,
  imageDomain = ref )
tdgenfun <- mytd$generate( batchSize = 10 )

#
track <- regressionModel$fit_generator(
  generator = reticulate::py_iterator( tdgenfun ),
  steps_per_epoch = 1,
  epochs = myep  )

#####################
mytd2 <- randomImageTransformParametersBatchGenerator$new(
  imageList = list( images[[ leaveout ]] ),
  transformType = "Affine",
  sdTransform = sdt,
  imageDomain = ref )
tdgenfun2 <- mytd2$generate( batchSize = 1 )
#####################
# generate new data #
#####################
rr = readAntsrTransform( reg$fwdtransforms[1] )
domainMask = ref * 0 + 1
for ( it in 1:10 ) {
  testpop <- tdgenfun2()
  k = 1
  testimg = makeImage( domainMask, testpop[[1]][k,,,1] )
  predictedData <- regressionModel %>% predict( testpop[[1]], verbose = 0 )
  # we are learning the mapping away from the template so now invert the solution
  affTx = createAntsrTransform( "AffineTransform", dimension = 2 )
  setAntsrTransformFixedParameters( affTx,
    getAntsrTransformFixedParameters(rr)*(1))
  setAntsrTransformParameters( affTx, predictedData[k,] )
#  setAntsrTransformParameters( affTx, testpop[[2]][k,] ) # true
  ####
  affTxI = invertAntsrTransform( affTx )
  learned = applyAntsrTransform( affTxI,  testimg, ref )
  reg = antsRegistration( ref, testimg, 'Affine' )
  cat("*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*\n")
  print( paste( "ref-test", antsImageMutualInformation( ref, testimg, nBins=16) ) )
  print( paste( "ref-lern", antsImageMutualInformation( ref, learned, nBins=16 ) ) )
  print( paste( "ref-reg", antsImageMutualInformation( ref, reg$warpedmovout, nBins=16)) )
  plot( testimg, doCropping=F, alpha = 0.5  )
  plot( reg$warpedmovout, doCropping=F, alpha = 0.5  )
  plot( learned, doCropping=F, alpha = 0.5  )
  }
