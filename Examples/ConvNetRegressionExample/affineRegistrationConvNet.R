library( ANTsRNet )
library( ANTsR )
library( abind )
library( keras )
#########################################################
imageIDs <- c( "r16", "r27", "r30", "r62", "r64", "r85" )
if ( ! exists( "scl" ) ) scl = 4
leaveout = 4
shapeSD = 10
if ( ! exists( "myep" ) ) myep = 25 # reasonable default
ref = ri( 16 )  %>% resampleImage( scl ) %>% iMath("Normalize")

images <- list()
priorParams = matrix( nrow = length( imageIDs ), ncol = 6 )
for( j in 1:25 )
  {
  i = j %% 6
  if ( i == 0 ) i = 1
  cat( "Processing image", imageIDs[i], "\n" )
  img  = antsImageRead( getANTsRData( imageIDs[i] ) )
  reg = antsRegistration( ref, img, "Affine", affIterations=c(100,50,20) )
  images[[i]] <- ( iMath( reg$warpedmovout, "Normalize" ) * 255 ) %>%
    resampleImage( scl )  %>% iMath("Normalize")
  priorParams[i,] = getAntsrTransformParameters(
    readAntsrTransform( reg$fwdtransforms[1] ) )
  }

affTx = createAntsrTransform( "AffineTransform", dimension = 2 )
numRegressors = length( getAntsrTransformParameters( affTx ) )
input_shape <- c( dim( images[[1]]), 1)

affmns = c( 1, 0, 0, 1, 0, 0 )
affcov = cov( priorParams ) * shapeSD
affcov[ 1:6,5:6 ] = 0
affcov[ 5:6,1:6 ] = 0
affcov[ 5,5 ] = 1e-2
affcov[ 6,6] = 1e-2
mytd <- randomImageTransformParametersBatchGenerator$new(
  imageList = images[ -leaveout ],
  transformType = "Affine",
  txParamMeans = affmns,
  txParamSDs = diag(6),
  imageDomain = ref )
tdgenfun <- mytd$generate( batchSize = 32 )
regressionModel <- createAlexNetModel2D( input_shape,
  numRegressors, mode = 'regression'  )
regressionModel %>% compile(
    loss = "mse",
    optimizer = optimizer_adam( ),
    metrics = list("mean_absolute_error")
  )

track <- regressionModel %>% fit_generator(
    generator = reticulate::py_iterator( tdgenfun ),
    steps_per_epoch = 16,
    epochs = myep  )

#####################
mytd2 <- randomImageTransformParametersBatchGenerator$new(
  imageList = list( images[[ leaveout ]] ),
  transformType = "Affine",
  txParamMeans = affmns,
  txParamSDs = affcov,
  imageDomain = ref )
tdgenfun2 <- mytd2$generate( batchSize = 1 )
#####################
# generate new data #
#####################
rr = readAntsrTransform( reg$fwdtransforms[1] )
domainMask = ref * 0 + 1
for ( it in 1:3 ) {
  testpop <- tdgenfun2()
  k = 1
  testimg = makeImage( domainMask, testpop[[1]][k,,,1] )
#  testpop[[1]][k,,,1] = as.array( learned )
  predictedData <- regressionModel %>% predict( testpop[[1]], verbose = 0 )
  # we are learning the mapping away from the template so now invert the solution
  affTx = createAntsrTransform( "AffineTransform", dimension = 2 )
  setAntsrTransformFixedParameters( affTx,
    getAntsrTransformFixedParameters(rr)*(1))
  setAntsrTransformParameters( affTx, predictedData[k,] )
  ####
  affTxI = invertAntsrTransform( affTx )
  learned = applyAntsrTransform( affTxI,  testimg, ref )
  reg = antsRegistration( ref, testimg, 'Affine', affIterations=c(20,20,10) )
  cat("*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*<>*\n")
  # the learned transform maps to the template space ie ref
  print( paste( "ref-test", antsImageMutualInformation(ref, testimg ) ) )
  print( paste( "ref-lern", antsImageMutualInformation(ref, learned ) ) )
  print( paste( "ref-reg",  antsImageMutualInformation(ref, reg$warpedmovout)))
  plot( testimg - ref , doCropping=F, alpha = 0.5  )
  Sys.sleep(1)
  plot( learned - ref, doCropping=F, alpha = 0.5  )
  Sys.sleep(1)
  }
