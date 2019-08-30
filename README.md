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

*Lowe D. et al, Object Recognition from Local Scale-Invariant Features, Proceedings of the International Conference on Computer Vision, 1999, vol 2, 1150-1157.*

*Calonder M. et al, BRIEF: Binary Robust Independent Elementary Features, Computer Science, 2010, vol 6314, 778-792.*

## Software used
http://www.matthewtoews.com/fba/featExtract1.6.tar.gz}{http://www.matthewtoews.com/fba/featExtract1.6.tar.gz

</content>
</snippet>
