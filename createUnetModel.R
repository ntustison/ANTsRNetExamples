createUnetModel2D <- function( inputImageSize, 
                               numberOfClassificationLabels = 1,
                               layers = 1:4, 
                               lowestResolution = 32, 
                               kernelSize = c( 3, 3 ), 
                               poolSize = c( 2, 2 ), 
                               strides = c( 2, 2 )
                             )
{

if ( ! usePkg( "keras" ) )
  {
  stop( "Please install the keras package." )
  }

inputs <- layer_input( shape = c( inputImageSize, 1 ) )

# Encoding path  

encodingConvolutionLayers <- list()
for( i in 1:length( layers ) )
  {
  numberOfFilters <- lowestResolution * 2 ^ ( layers[i] - 1 )

  if( i == 1 )
    {
    conv <- inputs %>% layer_conv_2d( filters = numberOfFilters, kernel_size = kernelSize, activation = 'relu', padding = 'same' )
    } else {
    conv <- pool %>% layer_conv_2d( filters = numberOfFilters, kernel_size = kernelSize, activation = 'relu', padding = 'same' )
    }
  encodingConvolutionLayers[[i]] <- conv %>% layer_conv_2d( filters = numberOfFilters, kernel_size = kernelSize, activation = 'relu', padding = 'same' )
  
  if( i < length( layers ) )
    {
    pool <- encodingConvolutionLayers[[i]] %>% layer_max_pooling_2d( pool_size = poolSize, strides = strides )
    }
  }

# Decoding path 

outputs <- encodingConvolutionLayers[[length( layers )]]
for( i in 2:length( layers ) )
  {
  numberOfFilters <- lowestResolution * 2 ^ ( length( layers ) - layers[i] )    
  outputs <- layer_concatenate( list( outputs %>%  
    layer_conv_2d_transpose( filters = numberOfFilters, 
      kernel_size = kernelSize, strides = strides, activation = 'relu', padding = 'same' ),
    encodingConvolutionLayers[[length( layers ) - i + 1]] ),
    axis = 3
    )

  outputs <- outputs %>%
    layer_conv_2d( filters = numberOfFilters, kernel_size = kernelSize, activation = 'relu', padding = 'same'  )  
  }
outputs <- outputs %>% layer_conv_2d( filters = numberOfClassificationLabels, kernel_size = c( 1, 1 ), activation = 'softmax' )
  
unetModel <- keras_model( inputs = inputs, outputs = outputs )
unetModel %>% compile( loss = 'categorical_crossentropy',
  optimizer = optimizer_adam( lr = 0.00001 , decay = 1e-6 ),  
  metrics = c( 'accuracy' ) )

return( unetModel )
}

