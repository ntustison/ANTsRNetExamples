library( keras )

keras::backend()$clear_session()

source( "../../Models/createResNetModel.R" )

mnistData <- dataset_mnist()

numberOfLabels <- length( unique( mnistData$train$y ) )

X_train <- array( mnistData$train$x, dim = c( dim( mnistData$train$x ), 1 ) )
Y_train <- keras::to_categorical( mnistData$train$y, numberOfLabels )

# we add a dimension of 1 to specify the channel size
inputImageSize <- c( dim( mnistData$train$x )[2:3], 1 )

resNetModel <- createResNetModel2D( inputImageSize = inputImageSize, 
  numberOfClassificationLabels = numberOfLabels )

resNetModel %>% compile( loss = 'categorical_crossentropy',
  optimizer = optimizer_adam( lr = 0.0001 ),  
  metrics = c( 'categorical_crossentropy', 'accuracy' ) )

track <- resNetModel %>% fit( X_train, Y_train, epochs = 40, batch_size = 32, 
  verbose = 1, shuffle = TRUE, validation_split = 0.2 )

# Now test the model

X_test <- array( mnistData$test$x, dim = c( dim( mnistData$test$x ), 1 ) )
Y_test <- keras::to_categorical( mnistData$test$y, numberOfLabels )

testingMetrics <- resNetModel %>% evaluate( X_test, Y_test )
predictedData <- resNetModel %>% predict( X_test, verbose = 1 )
