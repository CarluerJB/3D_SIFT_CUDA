<snippet>
  <content>
  
# 3D SIFT GPU

SIFT is an algorithm introduced by David G.Lowe in 1999. 
This code is based on the work of Matthew Towes at École de technologie supérieure ÉTS.
This is a CUDA implémentation of the base code. 
There is also a fast descriptor computation using BRIEF and 2 other method introduced in the linked publication.

## Installation

You will need at least cmake 3.10 and cuda 10 installed on your computer. 
1. Download it !
2. Go in main directory and cmake the CMAKEList file. 
3. Once all the makefile are create use the make to finish the installation
4. The first make will download all needed library so it will take time.

## Usage

This algorithm is design to extract features from 3D volumes. The main format are accepted ( Nifti and Analyse format)
The algorith will automatically use the best GPU card on your computer. 

    ./featExtract [options] \<input image\> \<output features\>
  
		<input image>: nifti (.nii,.hdr,.nii.gz) or raw input volume (IEEE 32-bit float, little endian).
		<output features>: output file with features.
		[options]
		  -w         : output feature geometry in world coordinates, NIFTI qto_xyz matrix (default is voxel units).
		  -2+        : double input image size.
		  -2-        : halve input image size.
		  -d[0-9]    : set device id to be used, 0 mean no device so CPU version will be used.
		  -b         : Use the BRIEF descriptor format
		  -br        : Use the RRIEF descriptor format
		  -bn        : Use the NRRIEF descriptor format

## History

Aug 2, 2019 : Publication of CUDA 3D SIFT

## Credits

Jean-Baptiste CARLUER at École de technologie supérieure ÉTS.

## Publication

TODO: Write publication and link it here

## References

*Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, and L. D.Jackel, “Backpropagation applied to handwritten zip code recognition,”Neural computa-tion, vol. 1, no. 4, pp. 541–551, 1989.*
*C. G. Harris, M. Stephenset al., “A combined corner and edge detector.” inAlvey visionconference, vol. 15, no. 50.  Citeseer, 1988, pp. 10–5244.*
*T. Lindeberg, “Feature detection with automatic scale selection,”International journal ofcomputer vision, vol. 30, no. 2, pp. 79–116, 1998.*
*D. G. Lowe, “Object recognition from local scale-invariant features,”Proceedings of the 7thIEEE International Conference on Computer Vision, vol. 2, pp. 1150–1157 vol.2, 1999.*
*K. Mikolajczyk and C. Schmid, “Scale & affine invariant interest point detectors,”Inter-national journal of computer vision, vol. 60, no. 1, pp. 63–86, 2004.*
*A. Krizhevsky, I. Sutskever, and G. E. Hinton, “Imagenet classification with deep convolu-
tional neural networks,” in Advances in neural information processing systems, 2012, pp.
1097–1105.*
*D. H. Hubel and T. N. Wiesel, “Receptive fields and functional architecture of monkey
striate cortex,” The Journal of physiology, vol. 195, no. 1, pp. 215–243, 1968.*
*P. Scovanner, S. Ali, and M. Shah, “A 3-dimensional sift descriptor and its application to
action recognition,” in Proceedings of the 15th ACM international conference on Multime-
dia. ACM, 2007, pp. 357–360.*
*G. T. Flitton, T. P. Breckon, and N. M. Bouallagu, “Object recognition using 3d sift in
complex ct volumes.” in BMVC, no. 1, 2010, pp. 1–12.*
*W. Cheung and G. Hamarneh, “n-sift: n-dimensional scale invariant feature transform,”
IEEE Transactions on Image Processing, vol. 18, no. 9, pp. 2012–2021, 2009.*
*S. Allaire, J. J. Kim, S. L. Breen, D. A. Jaffray, and V. Pekar, “Full orientation invariance
and improved feature selectivity of 3d sift with application to medical image analysis,”
in 2008 IEEE computer society conference on computer vision and pattern recognition
workshops. IEEE, 2008, pp. 1–8.*
*M. Toews and W. M. Wells III, “Efficient and robust model-to-image alignment using 3d
scale-invariant features,” Medical Image Analysis, 2013.*
*M. Toews and W. Wells, “Sift-rank: Ordinal description for invariant feature correspon-
dence,” in 2009 IEEE Conference on Computer Vision and Pattern Recognition. IEEE,
2009, pp. 172–177.*
*M. Toews, W. M. Wells, and L. Zöllei, “A feature-based developmental model of the infant
brain in structural mri,” in International Conference on Medical Image Computing and
Computer-Assisted Intervention. Springer, Berlin, Heidelberg, 2012, pp. 204–211.*
*M. Toews, L. Zöllei, and W. M. W. III, “Feature-based alignment of volumetric multi-modal
images,” in Information Processing in Medical Imaging, 2013.*
*G. Gill, M. Toews, and R. R. Beichel, “Robust initialization of active shape models for lung
segmentation in ct scans: a feature-based atlas approach,” Journal of Biomedical Imaging,
vol. 2014, p. 13, 2014.*
*M. Toews, C. Wachinger, R. S. J. Estepar, and W. M. Wells, “A feature-based approach
to big data analysis of medical images,” in International Conference on Information Pro-
cessing in Medical Imaging. Springer, Cham, 2015, pp. 339–350.*
*C. Wachinger, M. Toews, G. Langs, W. Wells, and P. Golland, “Keypoint transfer seg-
mentation,” in International Conference on Information Processing in Medical Imaging.
Springer, Cham, 2015, pp. 233–245.*
*——, “Keypoint transfer for fast whole-body segmentation,” IEEE transactions on medical
imaging, 2018.*
*J. Bersvendsen, M. Toews, A. Danudibroto, W. M. Wells III, S. Urheim, R. S. J. Estépar,
and E. Samset, “Robust spatio-temporal registration of 4d cardiac ultrasound sequences,”
in Medical Imaging 2016: Ultrasonic Imaging and Tomography, vol. 9790. International
Society for Optics and Photonics, 2016, p. 97900F.*
*D. Ni, Y. Qu, X. Yang, Y. P. Chui, T.-T. Wong, S. S. Ho, and P. A. Heng, “Volumetric
ultrasound panorama based on 3d sift,” in International Conference on Medical Image
Computing and Computer-Assisted Intervention. Springer, 2008, pp. 52–60.*
*M. Toews and W. M. Wells III, “How are siblings similar? how similar are siblings?
large-scale imaging genetics using local image features,” in International Symposium on
Biomedical Imaging (ISBI). IEEE, 2016, pp. 847–850.*
*K. Kumar, M. Toews, L. Chauvin, O. Colliot, and C. Desrosiers, “Multi-modal brain
fingerprinting: a manifold approximation based framework,” NeuroImage, vol. 183, pp.
212–226, 2018.*
*L. Chauvin, K. Kumar, C. Desrosiers, J. De Guise, and M. Toews, “Diffusion orientation
histograms (doh) for diffusion weighted image analysis,” in Computational Diffusion MRI.
Springer, Cham, 2018, pp. 91–99.*
*L. Chauvin, K. Kumar, C. Desrosiers, J. De Guise, W. Wells, and M. Toews, “Analyz-
ing brain morphology on the bag-of-features manifold,” in International Conference on
Information Processing in Medical Imaging. Springer, Cham, 2019, pp. 45–56.*
*L. Chauvin, K. Kumar, C. Wachinger, M. Vangel, J. de Guise, C. Desrosiers, W. Wells,
M. Toews, A. D. N. Initiative et al., “Neuroimage signature from salient keypoints is highly
specific to individuals and shared by close relatives,” NeuroImage, p. 116208, 2019.*
*J. Luo, M. Toews, I. Machado, S. Frisken, M. Zhang, F. Preiswerk, A. Sedghi, H. Ding,
S. Pieper, P. Golland et al., “A feature-driven active framework for ultrasound-based
brain shift compensation,” in International Conference on Medical Image Computing and
Computer-Assisted Intervention. Springer, Cham, 2018, pp. 30–38.*
*I. Machado, M. Toews, J. Luo, P. Unadkat, W. Essayed, E. George, P. Teodoro, H. Car-
valho, J. Martins, P. Golland et al., “Non-rigid registration of 3d ultrasound for neuro-
surgery using automatic feature detection and matching,” International journal of com-
puter assisted radiology and surgery, vol. 13, no. 10, pp. 1525–1538, 2018.*
*S. Frisken, M. Luo, I. Machado, P. Unadkat, P. Juvekar, A. Bunevicius, M. Toews,
W. Wells, M. I. Miga, and A. J. Golby, “Preliminary results comparing thin-plate splines
with finite element methods for modeling brain deformation during neurosurgery using
intraoperative ultrasound,” in Medical Imaging 2019: Image-Guided Procedures, Robotic
Interventions, and Modeling, vol. 10951. International Society for Optics and Photonics,
2019, p. 1095120.*
*S. Frisken, M. Luo, P. Juvekar, A. Bunevicius, I. Machado, P. Unadkat, M. M. Bertotti,
M. Toews, W. M. Wells, M. I. Miga et al., “A comparison of thin-plate spline deforma-
tion and finite element modeling to compensate for brain shift during tumor resection,”
International journal of computer assisted radiology and surgery, pp. 1–11, 2019.*
*J. Luo, S. Frisken, I. Machado, M. Zhang, S. Pieper, P. Golland, M. Toews, P. Unadkat,
A. Sedghi, H. Zhou et al., “Using the variogram for vector outlier screening: application
to feature-based image registration,” International journal of computer assisted radiology
and surgery, vol. 13, no. 12, pp. 1871–1880, 2018.*
*Université Paris-Est Marne-la-Vallée (UPEM), “Architecture technique - le gpu,” 2008,
[Online; accessed September 12, 2019]. [Online]. Available: http://igm.univ-mlv.fr/~dr/
XPOSE2008/CUDA_GPGPU/arch_cpu_gpu.jpg*
*——, “Gpus et cpus - architecture des gpu et cpu,” 2008, [Online; accessed September 12,
2019]. [Online]. Available: http://igm.univ-mlv.fr/~dr/XPOSE2008/CUDA_GPGPU/
arch_cpu_gpu.jpg*
*M. P. T. Ojala and D. Harwood, “Performance evaluation of texture measures with classi-
fication based on kullback discrimination of distributions,” Proceedings of the 12th IAPR
International Conference on Pattern Recognition (ICPR), vol. 1, pp. 582–585, 1994.*
*——, “A comparative study of texture measures with classification based on feature dis-
tributions,” Pattern Recognition, vol. 29, pp. 51–59, 1996.*
*M. Calonder, V. Lepetit, C. Strecha, and P. Fua, “Brief: Binary robust independent el-
ementary features,” Proceedings of the 11th European Conference on Computer Vision:
Part IV, pp. 778–792, 2010.*
*M. Toews and W. Wells, “Efficient and robust model-to-image alignment using 3d scale-
invariant features,” Medical image analysis, vol. 17, 11 2012.*


## Software used
http://www.matthewtoews.com/fba/featExtract1.6.tar.gz}{http://www.matthewtoews.com/fba/featExtract1.6.tar.gz

</content>
</snippet>
