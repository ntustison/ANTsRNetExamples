################################################################################
################################################################################
library( ANTsRNet )
library( ANTsR )
library( abind )
library( keras )

scl = 2
nc = 4
sm = 0.0
leaveout = 5
sdt = 0.15
bst = 1
txtype = "DeformationBasis"
if ( ! exists( "myep" ) ) myep = 30 # reasonable default
ref = ri( 16 ) %>% resampleImage( scl )
if ( ! exists( "dpca") ) {
  images <- ri( "all" )
  wlist  <- list()
  for( i in 1:length( images ) )
    {
    cat( "Processing image", i, "\n" )
    img  = images[[i]]
    reg = antsRegistration( ref, img, "SyN", totalSigma = 0.5 )
    wlist[[ i ]] =  composeTransformsToField( ref, reg$fwd[1] )
    images[[i]] <- ( iMath( reg$warpedmovout, "Normalize" ) * 255 ) %>%
      resampleImage( scl )
    }

  mskpca = ref * 0 + 1
#  dpca = multichannelPCA( wlist, mskpca, k=10, pcaOption=50 )
  dpca = multichannelPCA( wlist, mskpca, pcaOption='fastICA' )
  }

build_model <- function( input_shape, num_regressors ) {

  # Define model
  myact='linear'
#  myact='relu'
  filtSz = c( 32, 64, 64, 32, 32 )
  filtSz = c( 16, 32, 64, max( input_shape ), 32 )
  filtSz = c( 64, 32, 16, 32, 64, 32 )
  dilrt = c( as.integer(2), as.integer(2) )
  dilrt = as.integer(1)
  model <- keras_model_sequential() %>%
    layer_conv_2d(filters = filtSz[1], kernel_size = c(3,3), activation = myact,
                  input_shape = input_shape, dilation_rate = dilrt ) %>%
    layer_conv_2d(filters = filtSz[2], kernel_size = c(3,3), activation = myact, dilation_rate = dilrt ) %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_dropout( rate = 0.0 ) %>%
    layer_conv_2d(filters = filtSz[3], kernel_size = c(3,3), activation = myact, dilation_rate = dilrt ) %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_dropout( rate = 0.0 ) %>%
    layer_conv_2d(filters = filtSz[4], kernel_size = c(3,3), activation = myact, dilation_rate = dilrt ) %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_dropout( rate = 0.0 ) %>%
    layer_conv_2d(filters = filtSz[5], kernel_size = c(3,3), activation = myact, dilation_rate = dilrt ) %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_dropout( rate = 0.0 ) %>%
    layer_flatten() %>%
    layer_dense(units = filtSz[6], activation = myact) %>%
    layer_dropout(rate = 0.0 ) %>%
    layer_dense(units = num_regressors )

  model %>% compile(
    loss = "mse",
#    optimizer = optimizer_rmsprop(  ),
    optimizer = optimizer_adam( amsgrad = TRUE ),
    metrics = list("mean_absolute_error")
  )

  model
}

basisw = dpca$pcaWarps
# multiply by magic number 100 b/c learning sensitive to scaling
for ( i in seq_len( length( basisw ) ) )
  basisw[[i]] = basisw[[i]] * 1
numRegressors = length( basisw )
input_shape <- c( dim( images[[1]]), images[[1]]@components )
regressionModel <- build_model(  input_shape, numRegressors   )

mytd <- randomImageTransformParametersBatchGenerator$new(
  imageList = images,
  transformType = "DeformationBasis",
  sdTransform = sdt,
  imageDomain = ref,
  spatialSmoothing = sm,
  numberOfCompositions = nc,
  deformationBasis = basisw,
  deformationBasisMeans = rep( 0, numRegressors ),
  deformationBasisSDs = rep( sdt, numRegressors ) )
tdgenfun <- mytd$generate( batchSize = 10 )

track <- regressionModel$fit_generator(
  generator = reticulate::py_iterator( tdgenfun ),
  steps_per_epoch = 5,
  epochs = myep  )

#####################
mytd2 <- randomImageTransformParametersBatchGenerator$new(
#  imageList = list( images[[ leaveout ]] ),
  imageList = images,
  transformType = "DeformationBasis",
  sdTransform = sdt,
  imageDomain = ref,
  spatialSmoothing = sm,
  numberOfCompositions = nc,
  deformationBasis = basisw,
  deformationBasisMeans = rep( 0, numRegressors ),
  deformationBasisSDs = rep( sdt, numRegressors ) )
tdgenfun2 <- mytd2$generate( batchSize = 1 )
#####################
# generate new data #
#####################
domainMask = ref * 0 + 1
for ( it in 1:2 ) {
  testpop <- tdgenfun2()
  k = 1
  testimg = makeImage( mskpca, testpop[[1]][k,,,1] )
  plot( testimg, doCropping = F )
  t1=Sys.time()
  predictedData <- regressionModel %>% predict( testpop[[1]], verbose = 0 )
  t2=Sys.time()
  reg = antsRegistration( testimg, ref, 'SyN' )
  t3=Sys.time()
  mycor = cor( as.numeric(predictedData), as.numeric( testpop[[2]] ))
  print( paste("speedup:",as.numeric(t3-t2)/as.numeric(t2-t1), 'cor',mycor))
  # we are learning the mapping from the template to the target image
  mytx  = basisWarp( basisw, predictedData * ( bst ), nc, sm )
  learned = applyAntsrTransformToImage( mytx,  ref, testimg  )
  cat("*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*\n")
  print( paste( "ref2tar", antsImageMutualInformation( testimg, ref, nBins=16)) )
  print( paste( "lrn2tar", antsImageMutualInformation( testimg, learned, nBins=16)) )
  print( paste( "reg2tar", antsImageMutualInformation( testimg, reg$warpedmovout, nBins=16)) )
  plot( testimg, doCropping=F, alpha = 0.5  )
  plot( reg$warpedmovout, doCropping=F, alpha = 0.5  )
  plot( learned, doCropping=F, alpha = 0.5  )
  }
