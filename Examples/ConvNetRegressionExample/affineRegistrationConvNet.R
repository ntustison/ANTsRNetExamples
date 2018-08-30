###################
library( ANTsRNet )
library( ANTsR )
library( abind )
library( keras )

# for plaidml
use_implementation(implementation = c("keras"))
use_backend(backend = 'plaidml' )

imageIDs <- c( "r16", "r27", "r30", "r62", "r64", "r85" )
if ( ! exists( "images" ) ) {
  images <- list()
  priorParams = matrix( nrow = length( imageIDs ), ncol = 6 )
}
scl = 2
leaveout = 4
sdt = 10
if ( ! exists( "myep" ) ) myep = 50 # reasonable default
ref = ri( 16 )
if ( length( images ) <= 1 )
  for( i in 1:length( imageIDs ) )
    {
    cat( "Processing image", imageIDs[i], "\n" )
    img  = antsImageRead( getANTsRData( imageIDs[i] ) )
    reg = antsRegistration( ref, img, "Affine" )
    images[[i]] <- ( iMath( reg$warpedmovout, "Normalize" ) * 255 ) %>%
      resampleImage( scl )
    priorParams[i,] = getAntsrTransformParameters(
      readAntsrTransform( reg$fwdtransforms[1] ) )
    }
ref = ri( 16 )  %>% resampleImage( scl )
domainMask = ref * 0 + 1

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
    loss = "mse",
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

affmns = c( 1, 0, 0, 1, 0, 0 )
affcov = cov( priorParams ) * sdt
affcov[ 1:6,5:6 ] = 0
affcov[ 5:6,1:6 ] = 0
affcov[ 5,5 ] = 1e-2
affcov[ 6,6] = 1e-2

#####################
mytd2 <- randomImageTransformParametersBatchGenerator$new(
  imageList = list( images[[ leaveout ]] ),
  transformType = "Affine",
  txParamMeans = affmns,
  txParamSDs = affcov,
  imageDomain = ref )
tdgenfun2 <- mytd2$generate( batchSize = 5 )
testpop <- tdgenfun2()
testimg = makeImage( domainMask, testpop[[1]][1,,,1] )
plot(testimg,doCropping=F)

mytd <- randomImageTransformParametersBatchGenerator$new(
  imageList = images[ -leaveout ],
  transformType = "Affine",
  txParamMeans = affmns,
  txParamSDs =  affcov,
  imageDomain = ref )
tdgenfun <- mytd$generate( batchSize = 10 )
#
track <- regressionModel$fit_generator(
  generator = reticulate::py_iterator( tdgenfun ),
  steps_per_epoch = 5,
  epochs = myep  )

#####################
# generate new data #
#####################
rr = readAntsrTransform( reg$fwdtransforms[1] )
for ( it in 1:10 ) {
  testpop <- tdgenfun2()
  testimg = makeImage( domainMask, testpop[[1]][1,,,1] )
  t1=Sys.time()
  predictedData <- regressionModel %>% predict( testpop[[1]], verbose = 0 )
  t2=Sys.time()
  reg = antsRegistration( ref, testimg, 'Affine', affIterations=c(10,10) )
  t3=Sys.time()
  print( paste("speedup:",as.numeric(t3-t2)/as.numeric(t2-t1) ))
  # we are learning the mapping away from the template so now invert the solution
  affTx = createAntsrTransform( "AffineTransform", dimension = 2 )
  setAntsrTransformFixedParameters( affTx,
    getAntsrTransformFixedParameters(rr)*(1))
  setAntsrTransformParameters( affTx, predictedData[1,] )
  setAntsrTransformParameters( affTx, testpop[[2]][1,] ) # true
  ####
  affTxI = invertAntsrTransform( affTx )
  learned = applyAntsrTransform( affTxI,  testimg, ref )
  cat("*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*\n")
  print( paste( "ref-idt", antsImageMutualInformation( ref, testimg)))
  print( paste( "ref-reg", antsImageMutualInformation( ref, reg$warpedmovout)))
  print( paste( "ref-lrn", antsImageMutualInformation( ref, learned)))
  plot( testimg, doCropping=F, alpha = 0.5  )
#  plot( reg$warpedmovout, doCropping=F, alpha = 0.5  )
#  plot( learned, doCropping=F, alpha = 0.5  )
  }
