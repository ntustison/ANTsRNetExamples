################################################################################
# example adapted from keras rstudio example ###################################
################################################################################
# download data from:  figshare five example modalities
#
set.seed( 1 ) # b/c we use resampling - we want to keep folds the same over runs
library( abind )
library( ANTsRNet )
library( ANTsR )
library(keras)
numRegressors = 12
normimg <-function( img, scl ) {
  temp = iMath( img  %>% iMath( "PadImage", 10 ), "Normalize" ) * 1 - 0.5
  resampleImage( temp, scl )
}
scl = 2
rdir = "/Users/stnava/Downloads/five/t1/"
fns = Sys.glob( paste0( rdir, "modt1*nii.gz" ) )
templatefn = "/Users/stnava/Downloads/five/t1/modt1_x130.nii.gz"
ww = which( fns == templatefn )
leaveout = c( caret::createDataPartition( 1:length(fns), p=0.1,
  list = T )$Resample1, ww ) # leave out template
wimgs = seq_len( length( fns ) )[ -leaveout ]
template = antsImageRead( templatefn )
ref = normimg( template, scl )
mskpca = ref * 0 + 1
if ( ! exists( "images" ) ) {
  images = list()
  ct = 1
  for( i in 1:length( fns ) )
    {
    cat( "Processing image-MI", i, "\n" )
    img = normimg( antsImageRead( fns[ i ] ), scl )
    images[[ct]] <- resampleImageToTarget( img, ref )
    ct = ct + 1
    }
  }

#### Parameterization ####
# input image dimensions
img_rows <- as.integer( nrow( ref ) )
img_cols <- as.integer( ncol( ref ) )
# color channels (1 = grayscale, 3 = RGB)
img_chns <- 1L

# number of convolutional filters to use
filters <- 64L

# convolution kernel size
num_conv <- 3L

latent_dim <- 2L
intermediate_dim <- 128L
epsilon_std <- 1.0

# training parameters
batch_size <- 20L # 100L
epochs <- 55L


#### Model Construction ####

original_img_size <- c(img_rows, img_cols, img_chns)

x <- layer_input(shape = c(original_img_size))

conv_1 <- layer_conv_2d(
  x,
  filters = img_chns,
  kernel_size = c(2L, 2L),
  strides = c(1L, 1L),
  padding = "same",
  activation = "relu"
)

conv_2 <- layer_conv_2d(
  conv_1,
  filters = filters,
  kernel_size = c(2L, 2L),
  strides = c(2L, 2L),
  padding = "same",
  activation = "relu"
)

conv_3 <- layer_conv_2d(
  conv_2,
  filters = filters,
  kernel_size = c(num_conv, num_conv),
  strides = c(1L, 1L),
  padding = "same",
  activation = "relu"
)

conv_4 <- layer_conv_2d(
  conv_3,
  filters = filters,
  kernel_size = c(num_conv, num_conv),
  strides = c(1L, 1L),
  padding = "same",
  activation = "relu"
)

flat <- layer_flatten(conv_4)
hidden <- layer_dense(flat, units = intermediate_dim, activation = "relu")

z_mean <- layer_dense(hidden, units = latent_dim)
z_log_var <- layer_dense(hidden, units = latent_dim)

sampling <- function(args) {
  z_mean <- args[, 1:(latent_dim)]
  z_log_var <- args[, (latent_dim + 1):(2 * latent_dim)]

  epsilon <- k_random_normal(
    shape = c(k_shape(z_mean)[[1]]),
    mean = 0.,
    stddev = epsilon_std
  )
  z_mean + k_exp(z_log_var) * epsilon
}

z <- layer_concatenate(list(z_mean, z_log_var)) %>% layer_lambda(sampling)

output_shape <- c(batch_size, 47L, 34L, filters)

decoder_hidden <- layer_dense(units = intermediate_dim, activation = "relu")
decoder_upsample <- layer_dense(units = prod(output_shape[-1]), activation = "relu")

decoder_reshape <- layer_reshape(target_shape = output_shape[-1])
decoder_deconv_1 <- layer_conv_2d_transpose(
  filters = filters,
  kernel_size = c(num_conv, num_conv),
  strides = c(1L, 1L),
  padding = "same",
  activation = "relu"
)

decoder_deconv_2 <- layer_conv_2d_transpose(
  filters = filters,
  kernel_size = c(num_conv, num_conv),
  strides = c(1L, 1L),
  padding = "same",
  activation = "relu"
)


decoder_deconv_3_upsample <- layer_conv_2d_transpose(
  filters = filters,
  kernel_size = c(3L, 3L),
  strides = c(2L, 2L),
  padding = "valid",
  activation = "relu"
)

decoder_mean_squash <- layer_conv_2d(
  filters = img_chns,
  kernel_size = c(2L, 2L),
  strides = c(1L, 1L),
  padding = "valid",
  activation = "sigmoid"
)

hidden_decoded <- decoder_hidden(z)
up_decoded <- decoder_upsample(hidden_decoded)
reshape_decoded <- decoder_reshape(up_decoded)
deconv_1_decoded <- decoder_deconv_1(reshape_decoded)
deconv_2_decoded <- decoder_deconv_2(deconv_1_decoded)
x_decoded_relu <- decoder_deconv_3_upsample(deconv_2_decoded)
x_decoded_mean_squash <- decoder_mean_squash(x_decoded_relu)

# custom loss function
vae_loss <- function(x, x_decoded_mean_squash) {
  x <- k_flatten(x)
  x_decoded_mean_squash <- k_flatten(x_decoded_mean_squash)
  xent_loss <- 1.0 * img_rows * img_cols *
    loss_binary_crossentropy(x, x_decoded_mean_squash)
  kl_loss <- -0.5 * k_mean(1 + z_log_var - k_square(z_mean) -
                           k_exp(z_log_var), axis = -1L)
  k_mean(xent_loss + kl_loss)
}

## variational autoencoder
vae <- keras_model(x, x_decoded_mean_squash)
vae %>% compile(optimizer = "rmsprop", loss = vae_loss )
summary(vae)

## build a digit generator that can sample from the learned distribution
gen_decoder_input <- layer_input(shape = latent_dim)
gen_hidden_decoded <- decoder_hidden(gen_decoder_input)
gen_up_decoded <- decoder_upsample(gen_hidden_decoded)
gen_reshape_decoded <- decoder_reshape(gen_up_decoded)
gen_deconv_1_decoded <- decoder_deconv_1(gen_reshape_decoded)
gen_deconv_2_decoded <- decoder_deconv_2(gen_deconv_1_decoded)
gen_x_decoded_relu <- decoder_deconv_3_upsample(gen_deconv_2_decoded)
gen_x_decoded_mean_squash <- decoder_mean_squash(gen_x_decoded_relu)
generator <- keras_model(gen_decoder_input, gen_x_decoded_mean_squash)


#### Data Preparation ####
x_train = array( dim =  c( length(wimgs), img_rows, img_cols, img_chns   ) )
x_test = array( dim =  c( length(leaveout), img_rows, img_cols, img_chns   ) )
ct = 1
# FIXME coding style
for ( i in leaveout ) {
  x_test[ ct, , , 1 ] = as.array( images[[i]] )
  ct = ct + 1
  }
ct = 1
for ( i in wimgs ) {
  x_train[ ct, , , 1 ] = as.array( images[[i]] )
  ct = ct + 1
  }

#### Model Fitting ####
vae %>% fit(
  x_train, x_train,
  shuffle = TRUE,
  epochs = epochs,
  batch_size = batch_size,
  validation_data = list(x_test, x_test)
)


## encoder: model to project inputs on the latent space
encoder <- keras_model(x, z_mean)

## build a digit generator that can sample from the learned distribution
gen_decoder_input <- layer_input(shape = latent_dim)
gen_hidden_decoded <- decoder_hidden(gen_decoder_input)
gen_up_decoded <- decoder_upsample(gen_hidden_decoded)
gen_reshape_decoded <- decoder_reshape(gen_up_decoded)
gen_deconv_1_decoded <- decoder_deconv_1(gen_reshape_decoded)
gen_deconv_2_decoded <- decoder_deconv_2(gen_deconv_1_decoded)
gen_x_decoded_relu <- decoder_deconv_3_upsample(gen_deconv_2_decoded)
gen_x_decoded_mean_squash <- decoder_mean_squash(gen_x_decoded_relu)
generator <- keras_model(gen_decoder_input, gen_x_decoded_mean_squash)



#### Visualizations ####

library(ggplot2)
library(dplyr)

## display a 2D plot of the digit classes in the latent space
x_test_encoded <- predict(encoder, x_test, batch_size = batch_size)

hist( x_test_encoded[,2] )

z_sample = matrix( c( -2, -2 ), ncol=2 )
predimg =  as.antsImage( predict(generator, z_sample)[1,,,1] )
plot( predimg)
