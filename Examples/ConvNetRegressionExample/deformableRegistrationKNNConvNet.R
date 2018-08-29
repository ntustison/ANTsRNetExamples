################################################################################
################################################################################
library( ANTsRNet )
library( ANTsR )
library(keras)
# use_implementation("tensorflow")
library(tensorflow)
# tfe_enable_eager_execution(device_policy = "silent") # for the future
# library(tfdatasets)
library( abind )
normimg <-function( img, scl ) {
  temp = iMath( img, "Normalize" ) - 0.5
  temp = iMath( img, "Normalize" ) * 255
  resampleImage( temp, scl )
}
scl = 2
nc = 4
sm = 0.5
leaveout = c( 1 )  # leave out the template
sdt = 0.33
if ( ! exists( "bst" ) ) bst =  1 # should do line search on this value
txtype = "DeformationBasis"
if ( ! exists( "myep" ) ) myep = 10 # reasonable default
ref = ri( 16 ) %>% resampleImage( scl )
if ( ! exists( "dpca") ) {
  inimages <- ri( "all" )
  images = list()
  wlist  <- list()
  ct = 1
  wimgs = seq_len( length( inimages ) )[-leaveout]
  for( i in c(wimgs,wimgs,wimgs,wimgs) )
    {
    cat( "Processing image-MI", i, "\n" )
    noiseimage = makeImage( ref*0+1, rnorm( prod(dim(ref)), 0, 0.05 ) )
    img  = normimg( inimages[[i]], scl ) + noiseimage
    reg = antsRegistration( ref, img, "SyN", flowSigma = sample( c(3,4,5) )[1],
      affSampling=sample( c( 16, 20, 24, 28, 32 ) )[1], verbose=F )
    wlist[[ ct ]] =  composeTransformsToField( ref, reg$fwd[1] )
    images[[ct]] <- reg$warpedmovout
    ct = ct + 1
    }

  mskpca = getMask( ref ) %>% iMath( "MD", 6 )
#  dpca = multichannelPCA( wlist, mskpca, pcaOption='pca' )
  print('begin decomposition')
  dpca = multichannelPCA( wlist, mskpca, k=40, pcaOption=55 )
  basisw = dpca$pcaWarps
  # for some decompositions, we multiply by a magic number
  # b/c learning is sensitive to scaling
  defScale = 10
  for ( i in seq_len( length( basisw ) ) )
    basisw[[i]] = basisw[[i]] / defScale
  dpca$pca$v = dpca$pca$v / defScale
# We need this because not all of the allowable decompositions are SVD-like.
#  dpca = multichannelPCA( wlist, mskpca, pcaOption='fastICA' )
  pcaReconCoeffs = matrix( nrow = length( wlist ), ncol = ncol(dpca$pca$v)  )
  for ( i in 1:length( wlist ) ) {
    wvec = multichannelToVector( wlist[[i]], mskpca )
    mdl = lm( wvec ~ 0 + dpca$pca$v )
    pcaReconCoeffs[ i,  ] = coefficients(mdl)
  }
  pcaReconCoeffsMeans = colMeans( pcaReconCoeffs )
  pcaReconCoeffsSD = apply( pcaReconCoeffs, FUN=sd, MARGIN=2 )
  print('end decomposition')
  mskpca = ref *  0 + 1 # reset to full domain for deep learning
  numRegressors = length( basisw )
  }

###
onm = paste0( 'regi', numRegressors, 'KNNregressionModel.h5' )
bnm = tools::file_path_sans_ext( onm )
if ( file.exists( onm ) ) {
  regressionModel <- load_model_hdf5( onm )
  basisw = list( )
  for ( k in 1:numRegressors ) basisw[[ k ]] = antsImageRead( paste0( bnm, '_basis',k,'.nii.gz' ) )
  pcaReconCoeffsMeans = as.numeric( read.csv(  paste0( bnm, 'mn.csv' )  )[,1] )
  pcaReconCoeffsSD = as.numeric( read.csv(  paste0( bnm, 'sd.csv' )  )[,1] )
  }


build_model <- function( input_shape, num_regressors, dilrt = 1,
  myact='linear', drate = 0.0 ) {
  filtSz = c( 32, 32, 32, 32, 32, 32 )
  filtSz = c( 16, 32, 64, max( input_shape ), 64, 32 )
  dilrt = as.integer( dilrt )
  model <- keras_model_sequential() %>%
    layer_conv_2d(filters = filtSz[1], kernel_size = c(3,3), activation = myact,
                  input_shape = input_shape, dilation_rate = dilrt ) %>%
    layer_conv_2d(filters = filtSz[2], kernel_size = c(3,3), activation = myact, dilation_rate = dilrt ) %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_dropout( rate = drate ) %>%
#    layer_batch_normalization() %>%
    layer_conv_2d(filters = filtSz[3], kernel_size = c(3,3), activation = myact, dilation_rate = dilrt ) %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_dropout( rate = drate ) %>%
#    layer_batch_normalization() %>%
    layer_conv_2d(filters = filtSz[4], kernel_size = c(3,3), activation = myact, dilation_rate = dilrt ) %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_dropout( rate = drate ) %>%
#    layer_batch_normalization() %>%
    layer_conv_2d(filters = filtSz[5], kernel_size = c(3,3), activation = myact, dilation_rate = dilrt ) %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_dropout( rate = drate ) %>%
#    layer_batch_normalization() %>%
    layer_flatten() %>%
    layer_dense(units = filtSz[6], activation = myact) %>%
    layer_dropout(rate = drate ) %>%
    layer_dense(units = num_regressors )

  model %>% compile(
    loss = "mse",
    optimizer = optimizer_adam( ),
    metrics = list("mean_absolute_error")
  )

  model
}

input_shape <- c( dim( images[[1]]), images[[1]]@components )
mymus = pcaReconCoeffsMeans
mysds = pcaReconCoeffsSD * sdt
mytd <- randomImageTransformParametersBatchGenerator$new(
  imageList = images,
  transformType = "DeformationBasis",
  imageDomain = ref,
  spatialSmoothing = sm,
  numberOfCompositions = nc,
  deformationBasis = basisw,
  deformationBasisMeans = mymus * 0,
  deformationBasisSDs = mysds )
tdgenfun <- mytd$generate( batchSize = 10 )

##################### generate from a new source anatomy
# reg = antsRegistration( ref, ri( leaveout[2] ), "SyN", totalSigma = 0.0 )
# newanat = normimg( reg$warpedmovout, scl )
newanat = normimg( ref, scl )
newanat = normimg(  ri( sample( 2:6 )[1] )  , scl ) # original data
mytd2 <- randomImageTransformParametersBatchGenerator$new(
  imageList = list( newanat ),
  transformType = "DeformationBasis",
  imageDomain = ref,
  spatialSmoothing = sm,
  numberOfCompositions = nc,
  deformationBasis = basisw,
  deformationBasisMeans = mymus * 1,
  deformationBasisSDs = mysds * 1)
tdgenfun2 <- mytd2$generate( batchSize = 1 )

testpop <- tdgenfun2()
testimg = makeImage( mskpca, testpop[[1]][1,,,1] )
# plot( testimg, doCropping = F )

###
  # read bases, means and SDs as well
  if ( !file.exists( onm ) ) {
  if ( ! exists( "doTrain" ) ) doTrain = TRUE else doTrain = FALSE
  if ( doTrain ) {
    if ( ! exists( "regressionModel" ) )
      regressionModel <- build_model(  input_shape, numRegressors   )
    for ( trn in 1:100 ) {
      track <- regressionModel$fit_generator(
        generator = reticulate::py_iterator( tdgenfun ),
        steps_per_epoch = 10,
        epochs = myep  )
      print( paste( 'saving', onm, 'at loop', trn ) )
      save_model_hdf5( regressionModel, onm )
      # FIXME - need to save the bases as well! they can change over irlba runs
#      deformationBasis = basisw,
       for ( k in 1:length( basisw ) ) antsImageWrite( basisw[[k]], paste0( bnm, '_basis',k,'.nii.gz' ) )
       write.csv( pcaReconCoeffsMeans, paste0( bnm, 'mn.csv'),  row.names=F )
       write.csv( pcaReconCoeffsSD, paste0( bnm, 'sd.csv'), row.names=F )
#      deformationBasisSDs = mysds * 1)
      }
    }
  }

#####################
# generate new data #
#####################
domainMask = ref * 0 + 1
for ( it in 1:1 ) {
  cat("*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*\n")
  testpop <- tdgenfun2()
  k = 1
  testimg = makeImage( mskpca, testpop[[1]][k,,,1] )
#  plot( testimg, doCropping = F )
  t1=Sys.time()
  predictedData <- regressionModel %>% predict( testpop[[1]], verbose = 0 )
#  predictedData = testpop[[2]] # best possible result
  t2=Sys.time()
  reg = antsRegistration( testimg, ref, 'SyN' )
  t3=Sys.time()
  mycor = cor( as.numeric(predictedData), as.numeric( testpop[[2]] ))
  print( paste("speedup:",as.numeric(t3-t2)/as.numeric(t2-t1), 'cor',mycor))
  # we are learning the mapping from the template to the target image
  print(paste( "ref2tar", antsImageMutualInformation( testimg, ref, nBins=16)) )
    mytx  = basisWarp( basisw, predictedData * ( bst ), nc, sm )
    learned = applyAntsrTransformToImage( mytx,  ref, testimg  )
    print(paste("lrn2tar", antsImageMutualInformation( testimg, learned, nBins=16)))
    print( paste( "reg2tar", antsImageMutualInformation( testimg, reg$warpedmovout, nBins=16)) )
#  plot( testimg, doCropping=F, alpha = 0.5  )
##  plot( reg$warpedmovout, doCropping=F, alpha = 0.5  )
#  plot( learned, doCropping=F, alpha = 0.5  )
  }
