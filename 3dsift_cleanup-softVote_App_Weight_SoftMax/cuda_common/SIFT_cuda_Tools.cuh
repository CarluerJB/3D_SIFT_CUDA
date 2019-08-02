#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <builtin_types.h>
#include "PpImage.h"
#include "FeatureIO.h"
#include "nifti1_io.h"
#include "MultiScale.h"
#include <math.h>



#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


#define is_pixel(xaxis, yaxis, zaxis, sizeX, sizeY, sizeZ) \
 ((0 <= (xaxis) && (xaxis) < sizeX) && \
  (0 <= (yaxis) && (yaxis) < sizeY) && \
  (0 <= (zaxis) && (zaxis) < sizeZ))


__device__ float sign(float x);

__host__ void detectExtrema4D_test_cuda(
    FEATUREIO &inputH,
    FEATUREIO &inputC,
    FEATUREIO &fioSumOfSign,
    LOCATION_VALUE_XYZ_ARRAY &lvaMinima,
    LOCATION_VALUE_XYZ_ARRAY &lvaMaxima,
    int best_device_id);

__global__ void d_detectExtrema4D_test(
    FEATUREIO inputC,
    FEATUREIO inputH,
    FEATUREIO output,
    int tile_size,
    int cache_size
);


__host__ int
blur_3d_simpleborders_CUDA_Shared_mem(
  FEATUREIO &fio1,
  FEATUREIO &fioTemp,
  FEATUREIO &fio2,
  int	iFeature,
  PpImage &ppImgFilter);

__global__ void conv3d_shared(float *input,
  float *output,
  float *pfFilter,//COPY DONE
  int sizeX,
  int sizeY,
  int sizeZ,
  int kernel_size,
  int kernel_radius,
  int tile_size,
  int cache_size);


__host__ int
blur_3d_simpleborders_CUDA_Row_Col_Shared_mem(
  FEATUREIO &fio1,
  FEATUREIO &fioTemp,
  FEATUREIO &fio2,
  int	iFeature,
  PpImage &ppImgFilter,
  int best_device_id);

__global__ void conv3d_shared_Row_R(
  FEATUREIO input,
  FEATUREIO output,
  float *pfFilter,
  int kernel_size,
  int kernel_radius,
  int tile_size,
  int cache_size);

__global__ void conv3d_shared_Col_R(
  FEATUREIO input,
  FEATUREIO output,
  float *pfFilter,
  int kernel_size,
  int kernel_radius,
  int tile_size,
  int cache_size);

__global__ void conv3d_shared_Depth_R(
  FEATUREIO input,
  FEATUREIO output,
  float *pfFilter,
  int kernel_size,
  int kernel_radius,
  int tile_size,
  int cache_size);

__host__ int
blur_3d_simpleborders_CUDA_3x1D_W_Rot_Shared_mem(
  FEATUREIO &fio1,
  FEATUREIO &fioTemp,
  FEATUREIO &fio2,
  int	iFeature,
  PpImage &ppImgFilter,
  int best_device_id);

__global__ void conv3d_shared_Row(
  FEATUREIO input,
  FEATUREIO output,
  float *pfFilter,
  int kernel_size,
  int kernel_radius,
  int cache_size);

__global__ void conv3d_shared_Col(
  FEATUREIO input,
  FEATUREIO output,
  float *pfFilter,
  int kernel_size,
  int kernel_radius,
  int cache_size);

__global__ void conv3d_shared_Depth(
  FEATUREIO input,
  FEATUREIO output,
  float *pfFilter,
  int kernel_size,
  int kernel_radius,
  int cache_size);

__host__ int
blur_3d_simpleborders_CUDA_row_size(
  FEATUREIO &fio1,
  FEATUREIO &fioTemp,
  FEATUREIO &fio2,
  int	iFeature,
  PpImage &ppImgFilter);

__global__ void conv3d_shared_Row_size(
  float *input,
  float *output,
  float *pfFilter,
  int size_x,
  int size_y,
  int size_z,
  int kernel_size,
  int kernel_radius,
  int tile_size,
  int cache_size);

__host__ int
blur_3d_simpleborders_CUDA_BLOCK_Shared_mem(
  FEATUREIO &fio1,
  FEATUREIO &fioTemp,
  FEATUREIO &fio2,
  int	iFeature,
  PpImage &ppImgFilter);

__global__ void conv3d_shared_Row_BLOCK(
  float *input,
  float *output,
  float *pfFilter,
  int size_x,
  int size_y,
  int size_z,
  int kernel_size,
  int kernel_radius,
  int tile_size,
  int cache_size);

__global__ void conv3d_shared_Col_BLOCK(
  float *input,
  float *output,
  float *pfFilter,
  int size_x,
  int size_y,
  int size_z,
  int kernel_size,
  int kernel_radius,
  int tile_size,
  int cache_size);

__global__ void conv3d_shared_Depth_BLOCK(
  float *input,
  float *output,
  float *pfFilter,
  int size_x,
  int size_y,
  int size_z,
  int kernel_size,
  int kernel_radius,
  int tile_size,
  int cache_size);

  __host__ int SubSampleInterpolateCuda(
    FEATUREIO &fioIn,
    FEATUREIO &fioOut,
    int best_device_id);

  __global__ void cudaSubSampleInterpolate(
    FEATUREIO input,
    FEATUREIO output,
    int tile_size,
    int cache_size);

__host__ int fioCudaMultSum(
    FEATUREIO &fioIn1,
		FEATUREIO &fioIn2,
		FEATUREIO &fioOut,
		const float &fMultIn2);

__global__ void CudaMultSum(
  FEATUREIO input1,
  FEATUREIO input2,
  FEATUREIO output,
  float fMultIn2);
