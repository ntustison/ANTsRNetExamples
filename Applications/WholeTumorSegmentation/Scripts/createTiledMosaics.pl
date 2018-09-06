# /usr/bin/perl -w



my $baseDir = '/Volumes/Untitled/DeepLearningClassification/';
my $dataDir = "${baseDir}/Nifti/";
my $outputDir = "${baseDir}/PngOutput/";

my @subjects = ( <${dataDir}/UVA*/gbm*/>, <${dataDir}/UVA*/GLIOMA*/> );

for( my $i = 0; $i < @subjects; $i++ )
  {
  my $subjectDir = ${subjects[$i]};
  my @comps = split( '/', $subjectDir );
  my $subjectId = ${comps[-1]};

  print "$subjectId\n";

  my $t1 = "${subjectDir}/T1.nii.gz";
  my $contrast = "${subjectDir}/T1xCONTRASTWarped.nii.gz";
  my $flair = "${subjectDir}/T1xFLAIRWarped.nii.gz";
  my $brainMask = "${subjectDir}/BrainMaskProbability.nii.gz";
  my $tumorMask = "${subjectDir}/TumorMaskProbability.nii.gz";

  if( ! -e $t1 || ! -e $contrast || ! -e $flair || ! -e $brainMask || ! -e $tumorMask )
    {
    next;
    }

  my $brainMaskT1Png = "${subjectDir}/BrainMaskProbabilityWithT1.png";
  my $tumorMaskContrastPng = "${subjectDir}/TumorMaskProbabilityWithContrast.png";
  my $tumorMaskFlairPng = "${subjectDir}/TumorMaskProbabilityWithFlair.png";

  if( -e $brainMaskT1Png && -e $tumorMaskContrastPng && -e $tumorMaskFlairPng )
    {
    next;
    }

  my $tmpMask = "${subjectDir}/tmpMask.nii.gz";
  my $tmpMask2 = "${subjectDir}/tmpMask2.nii.gz";
  my $tmpMaskRgb = "${subjectDir}/tmpMaskRgb.nii.gz";
  my $tmpScaled = "${subjectDir}/tmpScaled.nii.gz";

  # Do brain mask with T1
  `ThresholdImage 3 $brainMask $tmpMask 0 0.5 0 1`;
  `ImageMath 3 $tmpScaled TruncateImageIntensity $t1`;
  `ConvertScalarImageToRGB 3 $tmpMask $tmpMaskRgb none red none 0 1 0 255`;

  my @args = ( 'CreateTiledMosaic', '-i', $tmpScaled,
                                    '-r', $tmpMaskRgb,
                                    '-o', $brainMaskT1Png,
                                    '-a', 0.5,
                                    '-t', '-1x-1',
                                    '-d', 2,
                                    '-g', 0,
                                    '-f', '0x1',
                                    '-p', 'mask',
                                    '-s', '[3,mask,mask]',
                                    '-x', $tmpMask );
  system( @args ) == 0 || next;

  # Do tumor with flair
  `ThresholdImage 3 $tumorMask $tmpMask2 0 0.5 0 1`;
  `ImageMath 3 $tmpScaled TruncateImageIntensity $flair`;
  `ConvertScalarImageToRGB 3 $tmpMask2 $tmpMaskRgb none red none 0 1 0 255`;

     @args = ( 'CreateTiledMosaic', '-i', $tmpScaled,
                                    '-r', $tmpMaskRgb,
                                    '-o', $tumorMaskFlairPng,
                                    '-a', 0.35,
                                    '-t', '-1x-1',
                                    '-d', 2,
                                    '-g', 0,
                                    '-f', '0x1',
                                    '-p', 'mask',
                                    '-s', '[3,mask,mask]',
                                    '-x', $tmpMask );
  system( @args ) == 0 || die "Error.\n";

    # Do tumor with contrast
  `ImageMath 3 $tmpScaled TruncateImageIntensity $contrast`;

     @args = ( 'CreateTiledMosaic', '-i', $tmpScaled,
                                    '-r', $tmpMaskRgb,
                                    '-o', $tumorMaskContrastPng,
                                    '-a', 0.35,
                                    '-t', '-1x-1',
                                    '-d', 2,
                                    '-g', 0,
                                    '-f', '0x1',
                                    '-p', 'mask',
                                    '-s', '[3,mask,mask]',
                                    '-x', $tmpMask );
  system( @args ) == 0 || die "Error.\n";

  unlink( $tmpScaled );
  unlink( $tmpMask );
  unlink( $tmpMask2 );
  unlink( $tmpMaskRgb );

  }
