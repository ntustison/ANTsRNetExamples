baseDir=./
inputDir=${baseDir}/TrainingData/
templateDir=${baseDir}/TemplateTransforms/
outputDir=${baseDir}/TrainingDataExpanded/

mkdir -p $outputDir

images=( `ls ${inputDir}/*H1*.nii.gz` )

for (( i=0; i < ${#images[@]}; i++ ));
  do
    imageSource=${images[$i]}
    imageSourceRoot=`basename $imageSource`
    imageSourceRoot=${imageSourceRoot/H1_2D\.nii\.gz/}

    maskSource=${imageSource/H1/Mask}

    for (( j=0; j < ${#images[@]}; j++ ));
      do
        imageTarget=${images[$j]}
        imageTargetRoot=`basename $imageTarget`
        imageTargetRoot=${imageTargetRoot/H1_2D\.nii\.gz/}

        imageSourceWarped=${outputDir}/${imageTargetRoot}x${imageSourceRoot}H1_2D.nii.gz;
        maskSourceWarped=${outputDir}/${imageTargetRoot}x${imageSourceRoot}Mask_2D.nii.gz;

        antsApplyTransforms -d 2 -i $imageSource \
                                 -r $imageTarget \
                                 -o $imageSourceWarped \
                                 -n Linear \
                                 -t [${templateDir}/T_${imageTargetRoot}${j}Affine.txt,1] \
                                 -t ${templateDir}/T_${imageTargetRoot}${j}InverseWarp.nii.gz \
                                 -t ${templateDir}/T_${imageSourceRoot}${i}Warp.nii.gz \
                                 -t ${templateDir}/T_${imageSourceRoot}${i}Affine.txt \
                                 -v 1

        antsApplyTransforms -d 2 -i $maskSource \
                                 -r $imageTarget \
                                 -o $maskSourceWarped \
                                 -n GenericLabel[Linear] \
                                 -t [${templateDir}/T_${imageTargetRoot}${j}Affine.txt,1] \
                                 -t ${templateDir}/T_${imageTargetRoot}${j}InverseWarp.nii.gz \
                                 -t ${templateDir}/T_${imageSourceRoot}${i}Warp.nii.gz \
                                 -t ${templateDir}/T_${imageSourceRoot}${i}Affine.txt \
                                 -v 1

      done  
  done
