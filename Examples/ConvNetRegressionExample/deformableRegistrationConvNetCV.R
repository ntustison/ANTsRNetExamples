################################################################################
# download data from:  figshare five example modalities
#
set.seed( 1 ) # b/c we use resampling - we want to keep folds the same over runs
runExtraOpt = FALSE
library( abind )
library( ANTsRNet )
library( ANTsR )
library(keras)
# for plaidml
usePlaid=TRUE
if ( usePlaid ) {
  use_implementation(implementation = c("keras"))
  use_backend(backend = 'plaidml' )
}

numRegressors = 35
shapeSD = 0.125
algid='randPCA'
algid='fastICA'
# algid=500
# algid='eanat'
onm = paste0( 'regi', numRegressors, 'alg',algid, 'regressionModel.h5' )
bnm = tools::file_path_sans_ext( onm )

imageMetric <- function( optParams, fImg, mImg, basisw, metricIn,
                         pcaParams, ncomp, smoothing ) {
  whichk=1:length(pcaParams)
  pcaParams[ whichk ] = optParams
  wtxlong = basisWarp( basisw, pcaParams, ncomp, smoothing )
  warped = applyAntsrTransform( wtxlong, data = mImg,
    reference = fImg )
  antsrMetricSetMovingImage( metricIn, warped )
  antsrMetricInitialize( metricIn )
  metricVal = antsrMetricGetValue( metricIn )
  return( metricVal )
}

imageMetricLS <- function( u, pcaParamsIn, gradIn, fImg, mImg,
  basisw, metricIn, ncomp, smoothing ) {
  pcaParams = pcaParamsIn + gradIn * u
  wtxlong = basisWarp( basisw, pcaParams, ncomp, smoothing )
  warped = applyAntsrTransform( wtxlong, data = mImg,
    reference = fImg )
  antsrMetricSetMovingImage( metricIn, warped )
  antsrMetricInitialize( metricIn )
  metricVal = antsrMetricGetValue( metricIn )
  return( metricVal )
}

normimg <-function( img, scl ) {
  temp = iMath( img  %>% iMath( "PadImage", 10 ), "Normalize" ) - 0.5
  resampleImage( temp, scl )
}
scl = 2
nc = 4
sm = 1.0
rdir = "/Users/stnava/Downloads/ndgenSliceStudy/"
rdir = "/Users/stnava/Downloads/five/t1/"
fns = Sys.glob( paste0( rdir, "modt1*nii.gz" ) )
templatefn = "/Users/stnava/Downloads/five/t1/modt1_x130.nii.gz"
ww = which( fns == templatefn )
leaveout = c( caret::createDataPartition( 1:length(fns), p=0.5,
  list = T )$Resample1, ww ) # leave out template
template = antsImageRead( templatefn )
txtype = "DeformationBasis"
if ( ! exists( "myep" ) ) myep = c( 10, 10 ) # epochs, batchSize
ref = normimg( template, scl )
mskpca = ref * 0 + 1
if ( !file.exists(  onm  ) )  {
  inimages <- imageFileNames2ImageList( fns )
  images = list()
  wlist  <- list()
  ct = 1
  wimgs = seq_len( length( inimages ) )[ -leaveout ]
  for( i in wimgs )
    {
    cat( "Processing image-MI", i, "\n" )
    img = normimg( inimages[[i]], scl )
    reg = antsRegistration( ref, img, "SyNBold",
      flowSigma = sample(c(3,6))[1], verbose=F)
    wlist[[ ct ]] =  composeTransformsToField( ref, reg$fwd[1] )
    images[[ct]] <- reg$warpedmovout
    ct = ct + 1
    }

  mskpca = ref * 0 + 1
  print('begin decomposition')
  basisw = wlist
  dpca = multichannelPCA( wlist, mskpca, k=numRegressors,
      pcaOption='randPCA', verbose=TRUE )
  basisw = dpca$pcaWarps
  numRegressors = length( basisw )
  # for some decompositions, we multiply by a magic number
  # b/c learning is sensitive to scaling
  defScale = 10.0
  for ( i in seq_len( length( basisw ) ) )
    basisw[[i]] = ( basisw[[i]] / mean( abs( basisw[[i]] ) ) ) / defScale
  mskpca = ref *  0 + 1 # reset to full domain for deep learning
  newv = matrix( nrow = prod( dim(mskpca ) )* mskpca@dimension,
    ncol = numRegressors )
  for ( myk in 1:numRegressors )
    newv[ , myk ] = multichannelToVector( basisw[[myk]], mskpca )
# We need this because not all of the allowable decompositions are SVD-like.
#  dpca = multichannelPCA( wlist, mskpca, pcaOption='fastICA' )
  pcaReconCoeffs = matrix( nrow = length( wlist ), ncol = numRegressors  )
  for ( i in 1:length( wlist ) ) {
    wvec = multichannelToVector( wlist[[i]], mskpca )
    mdl = lm( wvec ~ 0 + newv )
    pcaReconCoeffs[ i,  ] = coefficients(mdl)
  }
  pcaReconCoeffsMeans = colMeans( pcaReconCoeffs )
  pcaReconCoeffsSD = cov( pcaReconCoeffs )
  for ( k in 1:length( basisw ) ) antsImageWrite( basisw[[k]], paste0( bnm, '_basis',k,'.nii.gz' ) )
  write.csv( pcaReconCoeffsMeans, paste0( bnm, 'mn.csv'),  row.names=F )
  write.csv( pcaReconCoeffsSD, paste0( bnm, 'sd.csv'), row.names=F )
  print('end decomposition')
  }

if ( file.exists( onm ) ) regressionModel <- load_model_hdf5( onm )
if ( file.exists(  paste0( bnm, 'mn.csv' ) ) ) {
  pcaReconCoeffsMeans = as.numeric( read.csv(  paste0( bnm, 'mn.csv' )  )[,1] )
  pcaReconCoeffsSD = read.csv(  paste0( bnm, 'sd.csv' )  )
  numRegressors = length( pcaReconCoeffsMeans )
  basisw = list( )
  for ( k in 1:numRegressors ) basisw[[ k ]] = antsImageRead( paste0( bnm, '_basis',k,'.nii.gz' ) )
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

numRegressors = length( basisw )
mymus = pcaReconCoeffsMeans * 0
mysds = data.matrix( pcaReconCoeffsSD ) * shapeSD

##################### generate reference anatomy
mytd2 <- randomImageTransformParametersBatchGenerator$new(
  imageList = list( ref ),
  transformType = "DeformationBasis",
  imageDomain = ref,
  spatialSmoothing = sm,
  numberOfCompositions = nc,
  deformationBasis = basisw,
  txParamMeans = mymus,
  txParamSDs = mysds )
tdgenfun2 <- mytd2$generate( batchSize = 1 )

testpop <- tdgenfun2()
testimg = makeImage( mskpca, testpop[[1]][1,,,1] )
plot( testimg, doCropping = F )

if ( !file.exists( onm ) )
  {
  input_shape <- c( dim( ref ), ref@components )
  if ( ! exists( "doTrain" ) ) doTrain = TRUE else doTrain = FALSE
  if ( doTrain ) {
    if ( ! exists( "regressionModel" ) )
      regressionModel <- build_model(  input_shape, numRegressors   )
    mytd <- randomImageTransformParametersBatchGenerator$new(
							     imageList = images,
							     transformType = "DeformationBasis",
							     imageDomain = ref,
							     spatialSmoothing = sm,
							     numberOfCompositions = nc,
                   deformationBasis = basisw,
                   txParamMeans = mymus,
                   txParamSDs = mysds )
    tdgenfun <- mytd$generate( batchSize = 10 )
    for ( trn in 1:2 ) {
      track <- regressionModel$fit_generator(
        generator = reticulate::py_iterator( tdgenfun ),
        steps_per_epoch = myep[2],
        epochs = myep[1]  )
      print( paste( 'saving', onm, 'at loop', trn ) )
      save_model_hdf5( regressionModel, onm )
      }
    }
  }


###################################
# test on fully left out new data #
###################################
testpop <- tdgenfun2()
for ( k in leaveout ) {
  cat("*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*\n")
  testimg = normimg( antsImageRead( fns[k] ), scl )
  testimg = resampleImageToTarget( testimg, ref )
  testimg[testimg == 0] = min( testimg )
  t0=Sys.time()
  reg = antsRegistration( testimg, ref, 'SyN' )
  t1=Sys.time()
  testpop[[1]][1,,,1] = as.array( testimg )
  metricx = "Mattes"
  metric = antsrMetricCreate( testimg, ref, type = metricx,
    sampling.strategy = "regular", sampling.percentage = 0.25, nBins=32 )
  antsrMetricInitialize( metric )
  t2=Sys.time()
  inp = ( regressionModel %>% predict( testpop[[1]], verbose = 0 ) )[1,]
  if ( runExtraOpt ) { # run an extra line search optimization on top
    myg4 = inp
    bestval = optimize( imageMetricLS, lower=-1, upper=1,
      metricIn = metric,
      pcaParamsIn = inp, gradIn = myg4,
      basisw=basisw, fImg=testimg, mImg=ref,
      ncomp=nc, smoothing=sm  )
    print( bestval )
    outp = inp + myg4 * bestval$minimum
    } else outp = inp
  mytx  = basisWarp( basisw, outp, nc, sm )
  learned = applyAntsrTransformToImage( mytx,  ref, testimg  )
  t3=Sys.time()
  dltime = as.numeric(t3-t2)
  print( paste("speedup:",as.numeric(t1-t0)/dltime ))
  # we are learning the mapping from the template to the target image
  print(paste( "ref2tar", antsImageMutualInformation( testimg, ref)) )
  print(paste("lrn2tar", antsImageMutualInformation( testimg, learned)))
  print( paste( "reg2tar", antsImageMutualInformation( testimg,
    reg$warpedmovout)) )
  plot( testimg, doCropping=F, alpha = 0.5  )
  Sys.sleep(1)
  plot( learned, doCropping=F, alpha = 0.5  )
  Sys.sleep(1)
#  plot( reg$warpedmovout, doCropping=F, alpha = 0.5  )
  }
