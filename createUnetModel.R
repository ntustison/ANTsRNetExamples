
dice_coefficient <- function( y_true, y_pred )
{
  smoothingFactor <- 1

  K <- backend()  
  y_true_f <- K$flatten( y_true )
  y_pred_f <- K$flatten( y_pred )
  intersection <- K$sum( y_true_f * y_pred_f ) 
  return( ( 2.0 * intersection + smoothingFactor ) /
    ( K$sum( y_true_f ) + K$sum( y_pred_f ) + smoothingFactor ) )
}
attr( dice_coefficient, "py_function_name" ) <- "dice_coefficient"

loss_dice_coefficient_error <- function( y_true, y_pred )
{
  return( -dice_coefficient( y_true, y_pred ) )
}
attr( loss_dice_coefficient_error, "py_function_name" ) <- "dice_coefficient_error"

createUnetModel2D <- function( inputImageSize, 
                               numberOfClassificationLabels = 1,
                               layers = 1:4, 
                               lowestResolution = 32, 
                               convolutionKernelSize = c( 3, 3 ), 
                               deconvolutionKernelSize = c( 2, 2 ), 
                               poolSize = c( 2, 2 ), 
                               strides = c( 2, 2 )
                             )
{

if ( ! usePkg( "keras" ) )
  {
  stop( "Please install the keras package." )
  }

inputs <- layer_input( shape = c( inputImageSize, numberOfClassificationLabels ) )

# Encoding path  

encodingConvolutionLayers <- list()
for( i in 1:length( layers ) )
  {
  numberOfFilters <- lowestResolution * 2 ^ ( layers[i] - 1 )

  if( i == 1 )
    {
    conv <- inputs %>% layer_conv_2d( filters = numberOfFilters, kernel_size = convolutionKernelSize, activation = 'relu', padding = 'same' )
    } else {
    conv <- pool %>% layer_conv_2d( filters = numberOfFilters, kernel_size = convolutionKernelSize, activation = 'relu', padding = 'same' )
    }
  encodingConvolutionLayers[[i]] <- conv %>% layer_conv_2d( filters = numberOfFilters, kernel_size = convolutionKernelSize, activation = 'relu', padding = 'same' )
  
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
  
  # Note that the kernel size is different for the transpose layer

  outputs <- layer_concatenate( list( outputs %>%  
    layer_conv_2d_transpose( filters = numberOfFilters, 
      kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same' ),
    encodingConvolutionLayers[[length( layers ) - i + 1]] ),
    axis = 3
    )

  outputs <- outputs %>%
    layer_conv_2d( filters = numberOfFilters, kernel_size = convolutionKernelSize, activation = 'relu', padding = 'same'  )  %>%
    layer_conv_2d( filters = numberOfFilters, kernel_size = convolutionKernelSize, activation = 'relu', padding = 'same'  )  
  }
outputs <- outputs %>% layer_conv_2d( filters = numberOfClassificationLabels, kernel_size = c( 1, 1 ), activation = 'sigmoid' )
  
unetModel <- keras_model( inputs = inputs, outputs = outputs )
unetModel %>% compile( loss = loss_dice_coefficient_error,
  optimizer = optimizer_adam( lr = 0.0001 ),  
  metrics = c( dice_coefficient ) )

return( unetModel )
}

  