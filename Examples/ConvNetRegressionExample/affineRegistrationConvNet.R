

library( ANTsRNet )
library( ANTsR )
library( abind )
library( keras )
imageIDs <- c( "r16", "r27", "r30", "r62", "r64", "r85" )
images <- list()
ref = ri( 16 )  %>% resampleImage( 2 )
for( i in 1:length( imageIDs ) )
  {
  cat( "Processing image", imageIDs[i], "\n" )
  img  = antsImageRead( getANTsRData( imageIDs[i] ) ) %>% resampleImage( 2 )
  reg = antsRegistration( ref, img, "Affine" )
  images[[i]] <- reg$warpedmovout
  }

build_model <- function( input_shape, num_regressors ) {

  # Define model
  model <- keras_model_sequential() %>%
    layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
                  input_shape = input_shape) %>%
    layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_dropout(rate = 0.25) %>%
    layer_flatten() %>%
    layer_dense(units = 128, activation = 'relu') %>%
    layer_dropout(rate = 0.5) %>%
    layer_dense(units = num_regressors )

  model %>% compile(
    loss = "mse",
#    optimizer = optimizer_rmsprop(),
    optimizer = optimizer_adam( lr = 0.0001 ),
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
  imageList = images,
  transformType = "Affine",
  sdTransform = 0.1,
  imageDomain = images[[1]] )
tdgenfun <- mytd$generate( batchSize = 5 )

#
track <- regressionModel$fit_generator(
  generator = reticulate::py_iterator( tdgenfun ),
  steps_per_epoch = 2,
  epochs = 12  )

#####################
tdgenfun2 <- mytd$generate( batchSize = 10 )
#####################
# generate new data #
#####################
testpop <- tdgenfun2()
domainMask = img * 0 + 1
k = 3
testimg = makeImage( domainMask, testpop[[1]][k,,,1] )
predictedData <- regressionModel %>% predict( testpop[[1]], verbose = 0 )
# we are learning the mapping away from the template so now invert the solution
affTx = createAntsrTransform( "AffineTransform", dimension = 2 )
setAntsrTransformParameters( affTx, testpop[[2]][k,] )
affTxI = invertAntsrTransform( affTx )
rr = readAntsrTransform( reg$fwdtransforms[1] )
setAntsrTransformFixedParameters( affTxI, getAntsrTransformFixedParameters(rr))
learned = applyAntsrTransform( affTxI,  testimg, ref )
plot( ref, testimg, doCropping=F, alpha = 0.5  )
plot( ref, learned, doCropping=F, alpha = 0.5  )
#
# now compare the predicted to the real
#
for ( row in 1:nrow( predictedData ) )
  print( paste(
    'cor:',cor( predictedData[row,], testpop[[2]][row,] ),
    'abs-err:',mean(abs(  predictedData[row,] - testpop[[2]][row,] ) ) ) )

for ( col in 1:ncol( predictedData ) )
  print( paste(
    'cor:',cor( predictedData[,col], testpop[[2]][,col] ),
    'abs-err:',mean(abs(  predictedData[,col] - testpop[[2]][,col] ) ) ) )
