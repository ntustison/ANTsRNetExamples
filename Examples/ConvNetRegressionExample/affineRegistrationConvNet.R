library( ANTsRNet )
library( ANTsR )
library( abind )
library( keras )

build_model <- function( input_shape, num_regressors, dilrt = 1,
  myact='linear', drate = 0.0 ) {
  filtSz = c( 32,  16, 16, 32 )
  dilrt = as.integer( dilrt )
  idim = 2
  ksz = rep(3,idim)
  psz = rep(2,idim)
  model <- keras_model_sequential() %>%
    layer_conv_2d(filters = filtSz[1], kernel_size = ksz, activation = myact,
                  input_shape = input_shape, dilation_rate = dilrt ) %>%
    layer_max_pooling_2d(pool_size = psz) %>%
    layer_conv_2d(filters = filtSz[3], kernel_size = ksz, activation = myact ) %>%
    layer_max_pooling_2d(pool_size = psz) %>%
#    layer_conv_2d(filters = filtSz[2], kernel_size = ksz, activation = myact ) %>%
    layer_locally_connected_2d(filters = filtSz[3], kernel_size = ksz, activation = myact ) %>%
#    layer_max_pooling_2d(pool_size = psz) %>%
    layer_flatten() %>%
    layer_dense(units = filtSz[ length(filtSz) ], activation = myact) %>%
    layer_dense(units = num_regressors )
  model
}

#########################################################
imageIDs <- c( "r16", "r27", "r30", "r62", "r64", "r85" )
if ( ! exists( "scl" ) ) scl = 2
shapeSD = 10.0
if ( ! exists( "myep" ) ) myep = 25 # reasonable default
ref = ri( 16 )  %>% resampleImage( scl ) %>% iMath("Normalize")

nsub = 16
if ( ! exists( "priorParams") ) {
  images <- list()
  priorParams = matrix( nrow = nsub, ncol = 6 )
  noiseimage <- function( domain  ) {
    makeImage( domain, rnorm( prod( dim(domain) ), 128, 1  ) ) %>%
      smoothImage( 1 )
  }
  for( j in 1:nsub )
    {
    i = j %% 6
    if ( i == 0 ) i = 6
    cat( "Processing image", imageIDs[i], "\n" )
    img  = antsImageRead( getANTsRData( imageIDs[i] ) )
    img = antsRegistration( ref, img, "Translation", affIterations=c(100,50,20) )$warpedmovout
    img = img + noiseimage( img * 0 + 1 )
    reg = antsRegistration( ref, img, "Affine", affIterations=c(100,50,20) )
    images[[j]] <- ( iMath( reg$warpedmovout, "Normalize" ) * 255 ) %>%
      resampleImage( scl )  %>% iMath("Normalize")
    priorParams[j,] = getAntsrTransformParameters(
      readAntsrTransform( reg$fwdtransforms[1] ) )
    }
  }
affTx = createAntsrTransform( "AffineTransform", dimension = 2 )
numRegressors = length( getAntsrTransformParameters( affTx ) )
input_shape <- c( dim( images[[1]]), 1)
if ( ! exists( "regressionModel" ) ) {
  regressionModel <- build_model(input_shape,numRegressors)
  regressionModel %>% compile(
      loss = "mse",
      optimizer = optimizer_adam( ),
      metrics = list("mean_absolute_error")
    )
  regressionModel <- multi_gpu_model( regressionModel )
  }
affmns = colMeans( priorParams )
affcov = cov( priorParams ) * shapeSD
affcov[ 1:6,5:6 ] = 0
affcov[ 5:6,1:6 ] = 0
affcov[ 5,5 ] = 1e-4
affcov[ 6,6] = 1e-4
mytd <- randomImageTransformParametersBatchGenerator$new(
  imageList = images,
  transformType = "Affine",
  txParamMeans = affmns,
  txParamSDs = affcov,
  imageDomain = ref,
  center=T )
tdgenfun <- mytd$generate( batchSize = 32 )

test = tdgenfun()
k=14;plot( as.antsImage( test[[1]][k,,,1] )*11 )

if ( doTrain )
  track <- regressionModel %>% fit_generator(
      generator = tdgenfun,
      steps_per_epoch = 4,
      epochs = 100  )

#####################
# generate new data #
#####################
rr = readAntsrTransform( reg$fwdtransforms[1] )
domainMask = ref * 0 + 1
for ( it in 1:3 ) {
  testpop <- tdgenfun()
  k = 1
  testimg = makeImage( domainMask, testpop[[1]][k,,,1] )
  predictedData <- regressionModel %>% predict( testpop[[1]], verbose = 0 )
  affTx = createAntsrTransform( "AffineTransform", dimension = 2 )
  setAntsrTransformFixedParameters( affTx,
    getAntsrTransformFixedParameters(rr)*(1))
  setAntsrTransformParameters( affTx, predictedData[k,] + affmns )
  ####
  affTxI = invertAntsrTransform( affTx )
  learned = applyAntsrTransformToImage( affTxI,  testimg, ref )
  reg = antsRegistration( ref, testimg, 'Affine', affIterations=c(20,20,10) )
  cat("*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*\n")
  # the learned transform maps to the template space ie ref
  print( paste( "ref-test", antsImageMutualInformation(ref, testimg ) ) )
  print( paste( "ref-lern", antsImageMutualInformation(ref, learned ) ) )
  print( paste( "ref-reg",  antsImageMutualInformation(ref, reg$warpedmovout)))
  plot( ref*100 , doCropping=F, alpha = 0.5  )
  plot( learned*100 , doCropping=F, alpha = 0.5  )
  Sys.sleep(1)
  }
