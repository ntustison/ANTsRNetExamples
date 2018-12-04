library( ANTsRNet )
library( ANTsR )
library( abind )
library( keras )
imageIDs <- c( "r16", "r27", "r30", "r62", "r64", "r85" )
# Perform simple 3-tissue segmentation.  For convenience we are going
segmentationLabels <- c( 1, 2, 3 )
numberOfLabels <- length( segmentationLabels )

images <- list()
kmeansSegs <- list()
#
for( j in 1:22 )
  {
  i = j %% 6
  if ( i == 0 ) i = 1
  cat( "Processing image", imageIDs[i], "\n" )
  img  = antsImageRead( getANTsRData( imageIDs[i] ) ) %>% resampleImage( 2 )
  images[[i]] <- list( img )
  kmeansSegs[[i]] <- thresholdImage( img, "Otsu", 3 )
  }
###
if ( ! exists( "newModel") ) newModel = T
if ( newModel )
unetModel <- createUnetModel2D( c( dim( images[[1]][[1]] ),1 ),
    numberOfLayers=2, numberOfFiltersAtBaseLayer = 4, dropoutRate = 0.0,
    numberOfOutputs = numberOfLabels + 1, mode = 'classification' )

mytd <- randomImageTransformBatchGenerator$new(
  imageList = images,
  outcomeImageList = kmeansSegs,
  transformType = "Affine",
  sdAffine = 0.1,
  normalization = 'standardize',
  imageDomain = images[[1]][[1]],
  toCategorical = TRUE )

tdgenfun <- mytd$generate( batchSize = 1 )
#
if ( newModel )
unetModel %>% compile(
      loss = "categorical_crossentropy",
      optimizer = optimizer_adam( ),
      metrics = list("categorical_crossentropy")
    )

track <- unetModel %>% fit_generator(
      generator = tdgenfun,
      steps_per_epoch = 4,
      epochs = 250 )

diceOverlap <- function( x,  y ) {
  ulabs = sort( unique( c( unique(x), unique(y) ) ) )
  dicedf = data.frame( labels = ulabs, dice = rep( NA, length( ulabs ) ) )
  for ( ct in 1:length(ulabs) ) {
    denom = sum( x == ulabs[ct] ) + sum( y == ulabs[ct] )
    dd = sum( x == ulabs[ct] & y == ulabs[ct] ) * 2
    dicedf[  ct, 'dice' ] = dd / denom
  }
  return( dicedf )
}
#####################
k=1
testpop <- tdgenfun()
domainMask = img * 0 + 1
testimg = makeImage( domainMask, testpop[[1]][k,,,1] )
plot( testimg )
predictedData <- unetModel %>% predict( testpop[[1]], verbose = 0 )
probabilityImagesArray <- decodeUnet( predictedData, img )
for ( t in 1:1 ) {
  testimg = makeImage( domainMask, testpop[[1]][k,,,1] )
  segmat = matrix( predictedData[k,,,], nrow=tail(dim(predictedData),1) )
  for ( tt in 1:4 )
    segmat[tt,] = probabilityImagesArray[[k]][[tt]][ domainMask == 1 ]
  segvec = apply( segmat[,], FUN=which.max, MARGIN=2 )
  segimg = makeImage( domainMask, segvec )
  segimggt = thresholdImage( testimg, "Otsu", 3 )
  plot( testimg, segimg-1, alpha=0.8 )
  print( diceOverlap( segimggt, segimg-1 ) )
  Sys.sleep(1)
  }

# this demonstrates the augmentation style visually
if ( FALSE ) {
  library( ANTsRNet )
  library( ANTsR )
  imageIDs <- c( "r16", "r27", "r30", "r62", "r64", "r85" )
  kmeansSegsDemo = list()
  imagesDemo = list()
  for( i in 1:length( imageIDs ) )
    {
    cat( "Processing image", imageIDs[i], "\n" )
    imgDemo  = antsImageRead( getANTsRData( imageIDs[i] ) )
    imagesDemo[[i]] <- list( imgDemo )
    kmeansSegsDemo[[i]] <- thresholdImage( imgDemo, "Otsu", 3 )
    }
  rand = randomImageTransformAugmentation( imgDemo, imagesDemo,
    kmeansSegsDemo, n = 32, typeOfTransform = 'Deformation', sdAffine = 0.2 )
  for ( jj in 1:length( rand$outputPredictorList ) ) {
    plot( rand$outputPredictorList[[jj]][[1]],
      rand$outputOutcomeList[[jj]], alpha=0.1, doCropping=FALSE )
    Sys.sleep( 3 )
  }
  # reduce / increase the value of sdAffine to create less/more variation
}
