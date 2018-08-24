

library( ANTsRNet )
library( ANTsR )
library( abind )
library( keras )
imageIDs <- c( "r16", "r27", "r30", "r62", "r64", "r85" )
images <- list()
for( i in 1:length( imageIDs ) )
  {
  cat( "Processing image", imageIDs[i], "\n" )
  img  = antsImageRead( getANTsRData( imageIDs[i] ) ) %>% resampleImage( 2 )
  images[[i]] <- ( img )
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
    optimizer = optimizer_rmsprop(),
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
  sdTransform = 0.05,
  imageDomain = images[[1]] )
tdgenfun <- mytd$generate( batchSize = 5 )

#
track <- regressionModel$fit_generator(
  generator = reticulate::py_iterator( tdgenfun ),
  steps_per_epoch = 5,
  epochs = 25  )

#####################
tdgenfun2 <- mytd$generate( batchSize = 10 )
testpop <- tdgenfun2()

domainMask = img * 0 + 1
k = 1
testimg = makeImage( domainMask, testpop[[1]][k,,,1] )
plot( testimg )
predictedData <- regressionModel %>% predict( testpop[[1]], verbose = 0 )
# now compare the predicted to the real
for ( row in 1:nrow( predictedData ) )
  print( paste(
    'cor:',cor( predictedData[row,], testpop[[2]][row,] ),
    'abs-err:',mean(abs(  predictedData[row,] - testpop[[2]][row,] ) ) ) )
