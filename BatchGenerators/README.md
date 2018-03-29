### Sample batch generators which demonstrate various data augmentation schemes

* __ssdBatchGenerator3D.R:__  Batch generator for creation of 3-D SSD models using a 3-D template-based data augmentation strategy.  

* __ssdBatchGenerator2D.R:__  Batch generator for creation of 2-D SSD models using a 3-D template-based data augmentation strategy.  3-D images and corresponding transforms to/from a specific template are specified
in the batch generator call.  2-D slices of the warped input 3-D images are randomly sampled based on the 
axis direction specified in the call.

* __unetBatchGenerator.R:__  Batch generator for use with 2-D U-net models using a 2-D template-based data augmentation strategy and the magick package.
