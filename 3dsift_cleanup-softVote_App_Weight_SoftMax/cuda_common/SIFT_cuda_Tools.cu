// ===============================
// AUTHOR     : CARLUER Jean-Baptiste
// CREATE DATE     : 17/03/19
// PURPOSE     : Internship at the ETS, optimisation CUDA of the 3D SIFT of Matthew Towes
// SPECIAL NOTES: 
// ===============================
// Change History:
//                  * ADD -- 3D convolution with rotation
//                  * ADD -- 3D convolution without rotation (better perfomance)
//                  * DEL -- 1*3D convolution (to long)
//                  * ADD -- dOG between two FEATUREIO
//                  * ADD -- sum of sign, extremum detection
//                  * ADD -- Subsample of FEATUREIO
//                  * MOD -- Better optimisation of memory for each function
//                  * MOD -- index code error for Subsample
//
//==================================

#include "SIFT_cuda_Tools.cuh"


// ------------------- GAUSSIAN CONVOLUTION ------------------- //

//blur_3d_simpleborders_CUDA_Shared_mem()
//
//Realise block of 10*10*10 convolution
// 1* 3D Gaussian FILTER
// /!\ Don't work since memory optimisation, should have been removed
//
__host__ int
blur_3d_simpleborders_CUDA_Shared_mem(
  FEATUREIO &fio1,
  FEATUREIO &fioTemp,
  FEATUREIO &fio2,
  int	iFeature,
  PpImage &ppImgFilter) {
    // Set Sizes
    int iDataSizeFloat = fio1.x*fio1.y*fio1.z*fio1.t*fio1.iFeaturesPerVector*sizeof(float);
    int kernel_size = ppImgFilter.Cols();
  	int kernel_radius = ppImgFilter.Cols() / 2;
  	assert( (kernel_size%2) == 1 );
    int tile_size = 10;//kernel_size;
    int cache_size = (tile_size + (kernel_radius * 2));
    dim3 dimGrid(ceil(fio1.x/double(tile_size)), ceil(fio1.y/double(tile_size)), ceil(fio1.z/double(tile_size)));
    dim3 dimBlock(tile_size,tile_size,tile_size);
    int dimCache = cache_size*cache_size*cache_size*sizeof(float);


    // Allocation of device memory + memcpy
    float *d_fioIn; // INPUT IMAGE
    cudaMalloc((void**)&d_fioIn, iDataSizeFloat); //alloc image
    float *array_h=static_cast<float *>(fio1.pfVectors);//get a 1d array of pixel float
    cudaMemcpy(d_fioIn, array_h, iDataSizeFloat, cudaMemcpyHostToDevice); //get the array to device image

    float *d_fioOut; // OUTPUT IMAGE
    cudaMalloc((float**)&d_fioOut, iDataSizeFloat); //alloc image
    cudaMemset(d_fioOut, 0, iDataSizeFloat); // Set all val to 0

    float *d_pfFilter; // FILTER DATAS
    cudaMalloc((void**)&d_pfFilter, sizeof(float)*kernel_size);
    float *pfFilter_h=(float*)ppImgFilter.ImageRow(0);
    cudaMemcpy(d_pfFilter, pfFilter_h, sizeof(float)*kernel_size, cudaMemcpyHostToDevice);


    // Launch Kernel Filter
    conv3d_shared<<<dimGrid, dimBlock, dimCache>>>(d_fioIn, d_fioOut, d_pfFilter, fio1.x, fio1.y, fio1.z, kernel_size, kernel_radius, tile_size, cache_size);
    cudaDeviceSynchronize();
    //gpuErrchk( cudaPeekAtLastError() );


    // Copyback image filtered
    cudaMemcpy(fio2.pfVectors, d_fioOut, iDataSizeFloat, cudaMemcpyDeviceToHost);


    // Free Allocated Space on Device
    cudaFree(d_fioIn);
    cudaFree(d_fioOut);
    cudaFree(d_pfFilter);

    return 1;
}

__global__ void conv3d_shared(
  float *input,
  float *output,
  float *pfFilter,
  int size_x,
  int size_y,
  int size_z,
  int kernel_size,
  int kernel_radius,
  int tile_size,
  int cache_size) {

  // Set total workspace for this area
  extern __shared__ float shared_data[];

  // Set origin block of pixel coordinates
  int bx = blockIdx.x * tile_size;
  int by = blockIdx.y * tile_size;
  int bz = blockIdx.z * tile_size;

  // Set seed pixels coordinates
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  // Generate xyz position for center pixel
  int x_pos = bx + tx;
  int y_pos = by + ty;
  int z_pos = bz + tz;


  // Set index of focus pixel for each "worker".
  int tile_id = tz * tile_size * tile_size + ty * tile_size + tx; // before kernel size


  // Only keep enought worker for cover the cache_size(xy) area
  if (tile_id < cache_size * cache_size) {

    // Give a new 2D index to the thread : between (0,0) and (cache_size,cache_size)
    int tileZ = (int)((double)tile_id / (cache_size)) % (cache_size);
    int tileY =  tile_id % (cache_size);

    // Move the block pixel to upper left limit (3d view)
    int input_z = bz + tileZ - kernel_radius;
    int input_y = by + tileY - kernel_radius;
    int input_x_root = bx - kernel_radius; // partial Z =» seed for zAxis
    int input_x;

    // Make the zAxis Grow up
    for (int stemLength = 0; stemLength < cache_size; stemLength ++) {
      input_x = input_x_root + stemLength;
      if (is_pixel(input_x, input_y, input_z, size_x, size_y, size_z)){
        shared_data[tileZ*cache_size*cache_size + tileY*cache_size + stemLength] = input[input_z*size_y*size_x + input_y*size_x + input_x];
      } else {
        shared_data[tileZ*cache_size*cache_size + tileY*cache_size + stemLength] = 0.0f;
      }
    }
  }

  // Wait for other thread to complete the job (and fill shared_data)
  __syncthreads();


  // thread can then focus on their own pixel
  // Convolution
  if (is_pixel(x_pos, y_pos, z_pos, size_x, size_y, size_z)) {
    float xyzOutputValue = 0.0f;
    for (int z = 0; z < kernel_size; z ++) {
      float yzOutputValue = 0.0f;
      for (int y = 0; y < kernel_size; y ++) {
        float zOutputValue = 0.0f;
        for (int x = 0; x < kernel_size; x ++) {
            zOutputValue +=
              shared_data[(tx + x) + (ty + y)*(cache_size) + (tz + z)*(cache_size)*(cache_size)] *
              pfFilter[x];
        }
        yzOutputValue += zOutputValue*pfFilter[y];
      }
       xyzOutputValue += yzOutputValue*pfFilter[z];
       output[z_pos*(size_y)*(size_x) + y_pos*(size_x) + x_pos] = xyzOutputValue;

    }
  }
}



//
// blur_3d_simpleborders_CUDA_Row_Col_Shared_mem()
//
// CUDA ROLLING : 3x 1D GAUSSIAN FILTER WITHOUT Rotation
//

__host__ int
blur_3d_simpleborders_CUDA_Row_Col_Shared_mem(
  FEATUREIO &fio1,
  FEATUREIO &fioTemp,
  FEATUREIO &fio2,
  int	iFeature,
  PpImage &ppImgFilter,
  int best_device_id) {
    assert(best_device_id!=0);
    cudaSetDevice(best_device_id);
    // Set Sizes
    int iDataSizeFloat = fio2.x*fio2.y*fio2.z*fio2.t*fio2.iFeaturesPerVector*sizeof(float);
    int kernel_size = ppImgFilter.Cols();
  	int kernel_radius = ppImgFilter.Cols() / 2;
  	assert( (kernel_size%2) == 1 );
    int tile_size = 10;
    int cache_size = (tile_size + (kernel_radius * 2));
    dim3 dimBlock(tile_size,tile_size,tile_size);
    int dimCache = cache_size*tile_size*tile_size*sizeof(float)*fio1.iFeaturesPerVector;
    dim3 dimGrid(ceil(fio1.x/double(tile_size)), ceil(fio1.y/double(tile_size)), ceil(fio1.z/double(tile_size)));

    // Allocation of device memory + memcpy
    float *d_pfFilter; // FILTER DATA
    cudaMalloc((void**)&d_pfFilter, sizeof(float)*kernel_size);
    float *pfFilter_h=(float*)ppImgFilter.ImageRow(0);
    cudaMemcpy(d_pfFilter, pfFilter_h, sizeof(float)*kernel_size, cudaMemcpyHostToDevice);

    // Launch Kernel Filter
    conv3d_shared_Row_R<<<dimGrid, dimBlock, dimCache>>>(fio1, fio2, d_pfFilter, kernel_size, kernel_radius, tile_size, cache_size);
    cudaDeviceSynchronize();

    conv3d_shared_Col_R<<<dimGrid, dimBlock, dimCache>>>(fio2, fio1, d_pfFilter, kernel_size, kernel_radius, tile_size, cache_size);
    cudaDeviceSynchronize();

    conv3d_shared_Depth_R<<<dimGrid, dimBlock, dimCache>>>(fio1, fio2, d_pfFilter, kernel_size, kernel_radius, tile_size, cache_size);
    cudaDeviceSynchronize();

    // Copyback image filtered
    cudaMemcpy(fio2.pfVectors, fio2.d_pfVectors, iDataSizeFloat, cudaMemcpyDeviceToHost);


    // Free Allocated Space on Device
    cudaFree(d_pfFilter);
    return 1;
}

__global__ void conv3d_shared_Row_R(
  FEATUREIO input,
  FEATUREIO output,
  float *pfFilter,
  int kernel_size,
  int kernel_radius,
  int tile_size,
  int cache_size) {

  // Set total workspace for this area
  extern __shared__ float shared_data[];

  // Set origin block of pixel coordinates
  int bx = blockIdx.x * tile_size;
  int by = blockIdx.y * tile_size;
  int bz = blockIdx.z * tile_size;

  // Set seed pixels coordinates
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  // Generate xyz position for center pixel
  int x_pos = bx + tx;
  int y_pos = by + ty;
  int z_pos = bz + tz;

  // Set index of focus pixel for each "worker".
  int tile_id = tz * cache_size * tile_size + ty * tile_size + tx; // before kernel size


  // Only keep enought worker for cover the cache_size(xy) area
  if (tile_id < tile_size * tile_size*input.iFeaturesPerVector) {

    // Give a new 2D index to the thread : between (0,0) and (tile_size,tile_size)
    int tileZ = (int)((double)tile_id / (tile_size)) % (tile_size);
    int tileY =  tile_id % (tile_size);

    // Move the block pixel to upper left limit (3d view)
    int input_z = bz + tileZ;
    int input_y = by + tileY;
    int input_x_root = bx - kernel_radius; // partial Z =» seed for zAxis
    int input_x;

    // Make the zAxis Grow up
    for (int stemLength = 0; stemLength < cache_size; stemLength ++) {
      input_x = input_x_root + stemLength;
      if (is_pixel(input_x, input_y, input_z, input.x, input.y, input.z)){
        shared_data[tileZ*cache_size*tile_size*input.iFeaturesPerVector + tileY*cache_size*input.iFeaturesPerVector + stemLength*input.iFeaturesPerVector] = input.d_pfVectors[input_z*input.y*input.x*input.iFeaturesPerVector + input_y*input.x*input.iFeaturesPerVector + input_x*input.iFeaturesPerVector];
      } else {
        shared_data[tileZ*cache_size*tile_size*input.iFeaturesPerVector + tileY*cache_size*input.iFeaturesPerVector + stemLength*input.iFeaturesPerVector] = 0.0f;
      }
    }
  }
  // Wait for other thread to complete the job (and fill shared_data)
  __syncthreads();

  // thread can then focus on their own pixel
  // Convolution
  if (is_pixel(x_pos, y_pos, z_pos, output.x, output.y, output.z)) {
    float xOutputValue = 0.0f;
    for (int x = 0; x < kernel_size; x ++) {
        xOutputValue +=
          shared_data[(tx + x)*input.iFeaturesPerVector + (ty)*(cache_size)*input.iFeaturesPerVector + (tz)*(cache_size)*(tile_size)*input.iFeaturesPerVector] *
          pfFilter[x];
    }

    output.d_pfVectors[z_pos*(output.x)*(output.y)*input.iFeaturesPerVector + y_pos*(output.x)*input.iFeaturesPerVector + x_pos*input.iFeaturesPerVector ] = xOutputValue;
  }
}

__global__ void conv3d_shared_Col_R(
  FEATUREIO input,
  FEATUREIO output,
  float *pfFilter,
  int kernel_size,
  int kernel_radius,
  int tile_size,
  int cache_size) {

  // Set total workspace for this area
  extern __shared__ float shared_data[];

  // Set origin block of pixel coordinates
  int bx = blockIdx.x * tile_size;
  int by = blockIdx.y * tile_size;
  int bz = blockIdx.z * tile_size;

  // Set seed pixels coordinates
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  // Generate xyz position for center pixel
  int x_pos = bx + tx;
  int y_pos = by + ty;
  int z_pos = bz + tz;


  // Set index of focus pixel for each "worker".
  int tile_id = tz * tile_size * tile_size + ty * tile_size + tx; // before kernel size


  // Only keep enought worker for cover the cache_size(xy) area
  if (tile_id < tile_size * tile_size*input.iFeaturesPerVector) {

    // Give a new 2D index to the thread : between (0,0) and (cache_size,cache_size)
    int tileY = (int)((double)tile_id / (tile_size)) % (tile_size);
    int tileX =  tile_id % (tile_size);

    // Move the block pixel to upper left limit (3d view)
    int input_z_root = bz - kernel_radius;
    int input_y = by + tileY;
    int input_x = bx + tileX; // partial Z =» seed for zAxis
    int input_z;

    // Make the zAxis Grow up
    for (int stemLength = 0; stemLength < cache_size; stemLength ++) {
      input_z = input_z_root + stemLength;
      if (is_pixel(input_x, input_y, input_z, input.x, input.y, input.z)){
        shared_data[stemLength*tile_size*tile_size*input.iFeaturesPerVector + tileY*tile_size*input.iFeaturesPerVector + tileX*input.iFeaturesPerVector] = input.d_pfVectors[input_z*input.y*input.x*input.iFeaturesPerVector + input_y*input.x*input.iFeaturesPerVector + input_x*input.iFeaturesPerVector];
      } else {
        shared_data[stemLength*tile_size*tile_size*input.iFeaturesPerVector + tileY*tile_size*input.iFeaturesPerVector + tileX*input.iFeaturesPerVector] = 0.0f;
      }
    }
  }

  // Wait for other thread to complete the job (and fill shared_data)
  __syncthreads();

  // thread can then focus on their own pixel
  // Convolution
  if (is_pixel(x_pos, y_pos, z_pos, output.x, output.y, output.z)) {
    float yOutputValue = 0.0f;
    for (int z = 0; z < kernel_size; z ++) {
        yOutputValue +=
          shared_data[tx*input.iFeaturesPerVector + ty*(tile_size)*input.iFeaturesPerVector + (tz+z)*(tile_size)*(tile_size)*input.iFeaturesPerVector] *
          pfFilter[z];
    }
       output.d_pfVectors[z_pos*(output.y)*(output.x)*input.iFeaturesPerVector + y_pos*(output.x)*input.iFeaturesPerVector + x_pos*input.iFeaturesPerVector] = yOutputValue;
  }
}

__global__ void conv3d_shared_Depth_R(
  FEATUREIO input,
  FEATUREIO output,
  float *pfFilter,
  int kernel_size,
  int kernel_radius,
  int tile_size,
  int cache_size) {
  //printf("TEST\n");
  // Set total workspace for this area
  extern __shared__ float shared_data[];

  // Set origin block of pixel coordinates
  int bx = blockIdx.x * tile_size;
  int by = blockIdx.y * tile_size;
  int bz = blockIdx.z * tile_size;

  // Set seed pixels coordinates
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  // Generate xyz position for center pixel
  int x_pos = bx + tx;
  int y_pos = by + ty;
  int z_pos = bz + tz;


  // Set index of focus pixel for each "worker".
  int tile_id = tz * tile_size * tile_size + ty * tile_size + tx; // before kernel size


  // Only keep enought worker for cover the cache_size(xy) area
  if (tile_id < tile_size * tile_size*input.iFeaturesPerVector) {

    // Give a new 2D index to the thread : between (0,0) and (cache_size,cache_size)
    int tileZ = (int)((double)tile_id / (tile_size)) % (tile_size);
    int tileX =  tile_id % (tile_size);

    // Move the block pixel to upper left limit (3d view)
    int input_z = bz + tileZ;
    int input_y_root = by - kernel_radius;
    int input_x = bx + tileX; // partial Z =» seed for zAxis
    int input_y;

    // Make the zAxis Grow up
    for (int stemLength = 0; stemLength < cache_size; stemLength ++) {
      input_y = input_y_root + stemLength;
      if (is_pixel(input_x, input_y, input_z, input.x, input.y, input.z)){
        shared_data[tileZ*tile_size*cache_size*input.iFeaturesPerVector + stemLength*tile_size*input.iFeaturesPerVector + tileX*input.iFeaturesPerVector] = input.d_pfVectors[input_z*input.y*input.x*input.iFeaturesPerVector + input_y*input.x*input.iFeaturesPerVector + input_x*input.iFeaturesPerVector];
      } else {
        shared_data[tileZ*tile_size*cache_size*input.iFeaturesPerVector + stemLength*tile_size*input.iFeaturesPerVector + tileX*input.iFeaturesPerVector] = 0.0f;
      }
    }
  }

  // Wait for other thread to complete the job (and fill shared_data)
  __syncthreads();


  // thread can then focus on their own pixel
  // Convolution
  if (is_pixel(x_pos, y_pos, z_pos, output.x, output.y, output.z)) {
    float zOutputValue = 0.0f;
    for (int y = 0; y < kernel_size; y ++) {
        zOutputValue +=
          shared_data[tx*input.iFeaturesPerVector + (ty+y)*(tile_size)*input.iFeaturesPerVector + tz*(tile_size)*(cache_size)*input.iFeaturesPerVector] *
          pfFilter[y];
    }
       output.d_pfVectors[z_pos*(output.y)*(output.x)*input.iFeaturesPerVector + y_pos*(output.x)*input.iFeaturesPerVector + x_pos*input.iFeaturesPerVector] = zOutputValue;
  }
}


//
// blur_3d_simpleborders_CUDA_3x1D_W_Rot_Shared_mem()
//
// CUDA ROLLING : 3x 1D GAUSSIAN FILTER WITH Rotation
//
__host__ int
blur_3d_simpleborders_CUDA_3x1D_W_Rot_Shared_mem(
  FEATUREIO &fio1,
  FEATUREIO &fioTemp,
  FEATUREIO &fio2,
  int	iFeature,
  PpImage &ppImgFilter,
  int best_device_id) {

    //Check Device parameters
    assert(best_device_id!=0);
    cudaSetDevice(best_device_id);

    // Set Sizes
    int iDataSizeFloat = fio1.x*fio1.y*fio1.z*fio1.t*fio1.iFeaturesPerVector*sizeof(float);
    int kernel_size = ppImgFilter.Cols();
  	int kernel_radius = ppImgFilter.Cols() / 2;
  	assert( (kernel_size%2) == 1 );
    int tile_size = 10;//kernel_size;
    int cache_size = (tile_size + (kernel_radius * 2));
    dim3 dimBlock(tile_size,tile_size,tile_size);
    int dimCache = cache_size*tile_size*tile_size*sizeof(float);


    // Allocation of device memory + memcpy
    float *d_pfFilter; // FILTER DATAS
    cudaMalloc((void**)&d_pfFilter, sizeof(float)*kernel_size);
    float *pfFilter_h=(float*)ppImgFilter.ImageRow(0);
    cudaMemcpy(d_pfFilter, pfFilter_h, sizeof(float)*kernel_size, cudaMemcpyHostToDevice);


    // Launch Kernel Filter
    dim3 dimGrid(ceil(fio1.x/double(tile_size)), ceil(fio1.y/double(tile_size)), ceil(fio1.z/double(tile_size)));
    conv3d_shared_Row<<<dimGrid, dimBlock, dimCache>>>(fio1, fio2, d_pfFilter, kernel_size, kernel_radius, cache_size);


    dim3 dimGridY(ceil(fio1.y/double(tile_size)), ceil(fio1.z/double(tile_size)), ceil(fio1.x/double(tile_size)));
    conv3d_shared_Col<<<dimGridY, dimBlock, dimCache>>>(fio2, fio1, d_pfFilter, kernel_size, kernel_radius, cache_size);


    dim3 dimGridZ(ceil(fio1.z/double(tile_size)), ceil(fio1.x/double(tile_size)), ceil(fio1.y/double(tile_size)));
    conv3d_shared_Depth<<<dimGridZ, dimBlock, dimCache>>>(fio1, fio2, d_pfFilter, kernel_size, kernel_radius, cache_size);

    // Copyback image filtered
    cudaMemcpy(fio2.pfVectors, fio2.d_pfVectors, iDataSizeFloat, cudaMemcpyDeviceToHost);


    // Free Allocated Space on Device
    cudaFree(d_pfFilter);
    return 1;
}

__global__ void conv3d_shared_Row(
  FEATUREIO input,
  FEATUREIO output,
  float *pfFilter,
  int kernel_size,
  int kernel_radius,
  int cache_size) {

  // Set total workspace for this area
  extern __shared__ float shared_data[];

  // Set origin block of pixel coordinates
  int bx = blockIdx.x * blockDim.x;
  int by = blockIdx.y * blockDim.y;
  int bz = blockIdx.z * blockDim.z;

  // Set seed pixels coordinates
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  // Generate xyz position for center pixel
  int x_pos = bx + tx;
  int y_pos = by + ty;
  int z_pos = bz + tz;


  // Set index of focus pixel for each "worker".
  int tile_id = (tz * cache_size * blockDim.x) + (ty * blockDim.x) + tx; // before kernel size


  // Only keep enought worker for cover the cache_size(xy) area
  if (tile_id < blockDim.x * blockDim.y) {

    // Give a new 2D index to the thread : between (0,0) and (blockDim.y,blockDim.x)
    int tileZ = (int)((double)tile_id / (blockDim.y)) % (blockDim.x);
    int tileY =  tile_id % (blockDim.x);

    // Move the block pixel to upper left limit (3d view)
    int input_z = bz + tileZ;
    int input_y = by + tileY;
    int input_x_root = bx - kernel_radius; // partial Z =» seed for zAxis
    int input_x;

    // Make the zAxis Grow up
    for (int stemLength = 0; stemLength < cache_size; stemLength ++) {
      input_x = input_x_root + stemLength;
      if (is_pixel(input_x, input_y, input_z, input.x, input.y, input.z)){
        shared_data[tileZ*cache_size*blockDim.y + tileY*cache_size + stemLength] = input.d_pfVectors[input_x*input.y*input.z + input_z*input.y + input_y];
      } else {
        shared_data[tileZ*cache_size*blockDim.y + tileY*cache_size + stemLength] = 0.0f;
      }
    }
  }

  // Wait for other thread to complete the job (and fill shared_data)
  __syncthreads();

  // thread can then focus on their own pixel
  // Convolution
  if (is_pixel(x_pos, y_pos, z_pos, output.x, output.y, output.z)) {
    float xOutputValue = 0.0f;
    for (int x = 0; x < kernel_size; x ++) {
        xOutputValue +=
          shared_data[(tx + x) + (ty)*(cache_size) + (tz)*(cache_size)*(blockDim.y)] *
          pfFilter[x];
    }

    output.d_pfVectors[(x_pos)*(output.y)*(output.z) + (z_pos)*(output.y) + y_pos] = xOutputValue;
  }
}

__global__ void conv3d_shared_Col(
  FEATUREIO input,
  FEATUREIO output,
  float *pfFilter,
  int kernel_size,
  int kernel_radius,
  int cache_size) {

  // Set total workspace for this area
  extern __shared__ float shared_data[];

  // Set origin block of pixel coordinates
  int bx = blockIdx.x * blockDim.x;
  int by = blockIdx.y * blockDim.y;
  int bz = blockIdx.z * blockDim.z;

  // Set seed pixels coordinates
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  // Generate xyz position for center pixel
  int x_pos = bx + tx;
  int y_pos = by + ty;
  int z_pos = bz + tz;


  // Set index of focus pixel for each "worker".
  int tile_id = (tz * cache_size * blockDim.x) + (ty * blockDim.x) + tx; // before kernel size


  // Only keep enought worker for cover the cache_size(xy) area
  if (tile_id < blockDim.x * blockDim.y) {

    // Give a new 2D index to the thread : between (0,0) and (blockDim.y,blockDim.x)
    int tileZ = (int)((double)tile_id / (blockDim.y)) % (blockDim.x);
    int tileY =  tile_id % (blockDim.x);

    // Move the block pixel to upper left limit (3d view)
    int input_z = bz + tileZ;
    int input_y = by + tileY;
    int input_x_root = bx - kernel_radius; // partial Z =» seed for zAxis
    int input_x;

    // Make the zAxis Grow up
    for (int stemLength = 0; stemLength < cache_size; stemLength ++) {
      input_x = input_x_root + stemLength;
      if (is_pixel(input_x, input_y, input_z, input.y, input.z, input.x)){
        shared_data[tileZ*cache_size*blockDim.y + tileY*cache_size + stemLength] = input.d_pfVectors[input_y*input.x*input.y + input_x*input.x + input_z];
      } else {
        shared_data[tileZ*cache_size*blockDim.y + tileY*cache_size + stemLength] = 0.0f;
      }
    }
  }

  // Wait for other thread to complete the job (and fill shared_data)
  __syncthreads();


  // thread can then focus on their own pixel
  // Convolution
  if (is_pixel(x_pos, y_pos, z_pos, output.y, output.z, output.x)) {
    float xOutputValue = 0.0f;
    for (int x = 0; x < kernel_size; x ++) {
        xOutputValue +=
          shared_data[(tx + x) + (ty)*(cache_size) + (tz)*(cache_size)*(blockDim.y)] *
          pfFilter[x];
    }

    output.d_pfVectors[(y_pos)*(output.y)*(output.x) + (x_pos)*(output.x) + z_pos] = xOutputValue;
  }
}

__global__ void conv3d_shared_Depth(
  FEATUREIO input,
  FEATUREIO output,
  float *pfFilter,
  int kernel_size,
  int kernel_radius,
  int cache_size) {

  // Set total workspace for this area
  extern __shared__ float shared_data[];

  // Set origin block of pixel coordinates
  int bx = blockIdx.x * blockDim.x;
  int by = blockIdx.y * blockDim.y;
  int bz = blockIdx.z * blockDim.z;

  // Set seed pixels coordinates
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  // Generate xyz position for center pixel
  int x_pos = bx + tx;
  int y_pos = by + ty;
  int z_pos = bz + tz;


  // Set index of focus pixel for each "worker".
  int tile_id = (tz * cache_size * blockDim.x) + (ty * blockDim.x) + tx; // before kernel size


  // Only keep enought worker for cover the cache_size(xy) area
  if (tile_id < blockDim.x * blockDim.y) {

    // Give a new 2D index to the thread : between (0,0) and (blockDim.y,blockDim.x)
    int tileZ = (int)((double)tile_id / (blockDim.y)) % (blockDim.x);
    int tileY =  tile_id % (blockDim.x);

    // Move the block pixel to upper left limit (3d view)
    int input_z = bz + tileZ;
    int input_y = by + tileY;
    int input_x_root = bx - kernel_radius; // partial Z =» seed for zAxis
    int input_x;

    // Make the zAxis Grow up
    for (int stemLength = 0; stemLength < cache_size; stemLength ++) {
      input_x = input_x_root + stemLength;
      if (is_pixel(input_x, input_y, input_z, input.z, input.x, input.y)){
        shared_data[tileZ*cache_size*blockDim.y + tileY*cache_size + stemLength] = input.d_pfVectors[(input_z)*input.x*input.z + (input_y)*input.z + input_x];
      } else {
        shared_data[tileZ*cache_size*blockDim.y + tileY*cache_size + stemLength] = 0.0f;
      }
    }
  }

  // Wait for other thread to complete the job (and fill shared_data)
  __syncthreads();


  // thread can then focus on their own pixel
  // Convolution
  if (is_pixel(x_pos, y_pos, z_pos, output.z, output.x, output.y)) {
    float xOutputValue = 0.0f;
    for (int x = 0; x < kernel_size; x ++) {
        xOutputValue +=
          shared_data[(tx + x) + (ty)*(cache_size) + (tz)*(cache_size)*(blockDim.y)] *
          pfFilter[x];
    }

    output.d_pfVectors[(z_pos)*(output.x)*(output.z) + (y_pos)*(output.z) + x_pos] = xOutputValue;
  }
}


//
// BLOCK SIZE MAXIMISATION
//
// conv3d_shared_Depth_BLOCK()
//
// CUDA ROLLING : 3x 1D GAUSSIAN FILTER with block size maximisation (To LONG)
//
__host__ int
blur_3d_simpleborders_CUDA_BLOCK_Shared_mem(
  FEATUREIO &fio1,
  FEATUREIO &fioTemp,
  FEATUREIO &fio2,
  int	iFeature,
  PpImage &ppImgFilter){

    // cudaSetDevice(0); to select device
    // Set Sizes
    int iDataSizeFloat = fio1.x*fio1.y*fio1.z*fio1.t*fio1.iFeaturesPerVector*sizeof(float);
    int kernel_size = ppImgFilter.Cols();
  	int kernel_radius = ppImgFilter.Cols() / 2;
  	assert( (kernel_size%2) == 1 );
    int tile_size = 10;//kernel_size;
    int cache_size = (tile_size + (kernel_radius * 2));


    // Allocation of device memory + memcpy
    float *d_fioIn; // INPUT IMAGE
    cudaMalloc((void**)&d_fioIn, iDataSizeFloat); //alloc image
    float *array_h=static_cast<float *>(fio1.pfVectors);//get a 1d array of pixel float
    cudaMemcpy(d_fioIn, array_h, iDataSizeFloat, cudaMemcpyHostToDevice); //get the array to device image
    float *d_fioOut; // OUTPUT IMAGE
    cudaMalloc((float**)&d_fioOut, iDataSizeFloat); //alloc image
    cudaMemset(d_fioOut, 0, iDataSizeFloat); // Set all val to 0
    float *d_pfFilter; // FILTER DATAS
    cudaMalloc((void**)&d_pfFilter, sizeof(float)*kernel_size);
    float *pfFilter_h=(float*)ppImgFilter.ImageRow(0);
    cudaMemcpy(d_pfFilter, pfFilter_h, sizeof(float)*kernel_size, cudaMemcpyHostToDevice);


    // Launch Kernel Filter
    dim3 dimGrid(fio1.x, fio1.y, fio1.z);
    conv3d_shared_Row_BLOCK<<<dimGrid, kernel_size>>>(d_fioIn, d_fioOut, d_pfFilter, fio1.x, fio1.y, fio1.z, kernel_size, kernel_radius, tile_size, cache_size);
    cudaDeviceSynchronize();
    gpuErrchk( cudaPeekAtLastError() );

    cudaMemset(d_fioIn, 0.0, iDataSizeFloat); // Set all val to 0

    conv3d_shared_Col_BLOCK<<<dimGrid, kernel_size>>>(d_fioOut, d_fioIn, d_pfFilter, fio1.x, fio1.y, fio1.z, kernel_size, kernel_radius, tile_size, cache_size);
    cudaDeviceSynchronize();
    gpuErrchk( cudaPeekAtLastError() );

    cudaMemset(d_fioOut, 0.0, iDataSizeFloat); // Set all val to 0

    conv3d_shared_Depth_BLOCK<<<dimGrid, kernel_size>>>(d_fioIn, d_fioOut, d_pfFilter, fio1.x, fio1.y, fio1.z, kernel_size, kernel_radius, tile_size, cache_size);
    cudaDeviceSynchronize();
    gpuErrchk( cudaPeekAtLastError() );


    // Copyback image filtered
    cudaMemcpy(fio2.pfVectors, d_fioOut, iDataSizeFloat, cudaMemcpyDeviceToHost);


    // Free Allocated Space on Device
    cudaFree(d_fioIn);
    cudaFree(d_fioOut);
    cudaFree(d_pfFilter);
    return 1;
}

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
  int cache_size){

  // Set total workspace for this area
  //extern __shared__ float shared_data[];

  // Set origin block of pixel coordinates
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int bz = blockIdx.z;

  // Set seed pixels coordinates
  for(int tx_norm=-kernel_radius; tx_norm<kernel_radius; tx_norm++){
    int x_pos = bx+ tx_norm;
    int y_pos = by;
    int z_pos = bz;
    if (is_pixel(x_pos, y_pos, z_pos, size_x, size_y, size_z)){
      output[bz*size_x*size_y + by*size_x + bx] += input[z_pos*size_x*size_y + y_pos*size_x + x_pos]* pfFilter[tx_norm+kernel_radius];
    }
  }
}

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
  int cache_size){

  // Set total workspace for this area
  //extern __shared__ float shared_data[];

  // Set origin block of pixel coordinates
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int bz = blockIdx.z;

  // Set seed pixels coordinates
  for(int tx_norm=-kernel_radius; tx_norm<kernel_radius; tx_norm++){
    int x_pos = bx;
    int y_pos = by+ tx_norm;
    int z_pos = bz;
    if (is_pixel(x_pos, y_pos, z_pos, size_x, size_y, size_z)){
      output[bz*size_x*size_y + by*size_x + bx] += input[z_pos*size_x*size_y + y_pos*size_x + x_pos]* pfFilter[tx_norm+kernel_radius];
    }
  }
}

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
  int cache_size){

  // Set total workspace for this area
  //extern __shared__ float shared_data[];

  // Set origin block of pixel coordinates
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int bz = blockIdx.z;

  // Set seed pixels coordinates
  for(int tx_norm=-kernel_radius; tx_norm<kernel_radius; tx_norm++){
    int x_pos = bx;
    int y_pos = by;
    int z_pos = bz+ tx_norm;
    if (is_pixel(x_pos, y_pos, z_pos, size_x, size_y, size_z)){
      output[bz*size_x*size_y + by*size_x + bx] += input[z_pos*size_x*size_y + y_pos*size_x + x_pos]* pfFilter[tx_norm+kernel_radius];
    }
  }
}


//
// Row Col Depth SIZE Convolution (TO SLOW)
//
// blur_3d_simpleborders_CUDA_row_size()
//
// Realise a convolution with one thread for each pixel of one row *2 => col and depth not finish cause already very long
//
__host__ int
blur_3d_simpleborders_CUDA_row_size(
  FEATUREIO &fio1,
  FEATUREIO &fioTemp,
  FEATUREIO &fio2,
  int	iFeature,
  PpImage &ppImgFilter){
    int iDataSizeFloat = fio1.x*fio1.y*fio1.z*fio1.t*fio1.iFeaturesPerVector*sizeof(float);
    int kernel_size = ppImgFilter.Cols();
  	int kernel_radius = ppImgFilter.Cols() / 2;
  	assert( (kernel_size%2) == 1 );
    int tile_size = fio1.x;//kernel_size;
    int nb_row_treated = 2;
    //int cache_size = tile_size*3;
    int cache_size = (tile_size + (kernel_radius * 2));
    dim3 dimGrid(1, ceil(fio1.y/2), fio1.z);
    dim3 dimBlock(fio1.x, nb_row_treated, 1);

    //printf("DIMCACHE : %d, kernel_size : %d, tile_size : %d, kernel_radius : %d\n, cache_size : %d\n", dimCache, kernel_size, tile_size,kernel_radius,cache_size);
    //printf("IMAGE SIZE : (%d, %d, %d)\n", fio1.x, fio1.y, fio1.z);
    // Allocation of device memory + memcpy
    float *d_fioIn; // INPUT IMAGE
    cudaMalloc((void**)&d_fioIn, iDataSizeFloat); //alloc image
    float *array_h=static_cast<float *>(fio1.pfVectors);//get a 1d array of pixel float
    cudaMemcpy(d_fioIn, array_h, iDataSizeFloat, cudaMemcpyHostToDevice); //get the array to device image

    float *d_fioOut; // OUTPUT IMAGE
    cudaMalloc((float**)&d_fioOut, iDataSizeFloat); //alloc image
    cudaMemset(d_fioOut, 0, iDataSizeFloat); // Set all val to 0

    float *d_pfFilter; // FILTER DATAS
    cudaMalloc((void**)&d_pfFilter, sizeof(float)*kernel_size);
    float *pfFilter_h=(float*)ppImgFilter.ImageRow(0);
    cudaMemcpy(d_pfFilter, pfFilter_h, sizeof(float)*kernel_size, cudaMemcpyHostToDevice);
    // Launch Kernel Filter
    conv3d_shared_Row_size<<<dimGrid, dimBlock, (nb_row_treated*cache_size*sizeof(float))>>>(d_fioIn, d_fioOut, d_pfFilter, fio1.x, fio1.y, fio1.z, kernel_size, kernel_radius, tile_size, cache_size);
    cudaDeviceSynchronize();
    gpuErrchk( cudaPeekAtLastError() );
    /*conv3d_shared_Col<<<dimGrid, dimBlock, dimCache>>>(d_fioOut, d_fioIn, d_pfFilter, fio1.x, fio1.y, fio1.z, kernel_size, kernel_radius, tile_size, cache_size);
    cudaDeviceSynchronize();

    gpuErrchk( cudaPeekAtLastError() );
    conv3d_shared_Depth<<<dimGrid, dimBlock, dimCache>>>(d_fioIn, d_fioOut, d_pfFilter, fio1.x, fio1.y, fio1.z, kernel_size, kernel_radius, tile_size, cache_size);
    cudaDeviceSynchronize();
    gpuErrchk( cudaPeekAtLastError() );*/


    // Copyback image filtered
    cudaMemcpy(fio2.pfVectors, d_fioOut, iDataSizeFloat, cudaMemcpyDeviceToHost);


    // Free Allocated Space on Device
    cudaFree(d_fioIn);
    cudaFree(d_fioOut);
    cudaFree(d_pfFilter);
    return 1;
  }

// 18 sec just for row -> to slow
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
  int cache_size) {

  // Set total workspace for this area
  extern __shared__ float shared_data[];

  // Generate xyz position for center pixel
  int bx = threadIdx.x;
  int by = blockIdx.y * (blockDim.y);
  int bz = blockIdx.z;

  // Set seed pixels coordinates
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  //init shared memory to 0.0
  shared_data[tx + ty*size_x] = 0.0;
  if (tx<kernel_radius*2) {
    shared_data[tx + ty*size_x + (tx+1)] = 0.0;
  }

  // fill shared memory with a row of x
  shared_data[(tx + kernel_radius) + ty*size_x ] = input[bz * size_x * size_y + (by+ty)*size_x + bx];
  __syncthreads();


  // thread can then focus on their own pixel
  // Convolution
  if (is_pixel(bx, by, bz, size_x, size_y, size_z)) {
    float xOutputValue = 0.0f;
    for (int x = 0; x < kernel_size; x ++) {
        xOutputValue +=
          shared_data[(tx) + (x) + (ty)*(size_x) ] *
          pfFilter[x];
    }
       output[bz*size_x*size_y + (by+ty) * size_x + bx] = xOutputValue;
  }
}


// ------------------ Other Function that can be speed up with GPU------------------


//
// SubSampleInterpolateCuda()
//
// Subsample data using cuda, the parameter are set to the output
// this will lead to create 1000 pixel by block instead of 500 by block (if set to input)
//
__host__ int
SubSampleInterpolateCuda(
  FEATUREIO &fioIn,
  FEATUREIO &fioOut,
  int best_device_id){

  cudaSetDevice(best_device_id);
  int iDataSizeFloat = fioOut.x*fioOut.y*fioOut.z*fioOut.iFeaturesPerVector*sizeof(float);
  assert( fioIn.iFeaturesPerVector == fioOut.iFeaturesPerVector );

	if( fioIn.iFeaturesPerVector != fioOut.iFeaturesPerVector )
	{
		return 0;
	}

	if( fioOut.z > 1 )
	{
		assert( fioIn.x/2 == fioOut.x &&
			fioIn.y/2 == fioOut.y &&
			fioIn.z/2 == fioOut.z );
	}
	else
	{
		assert( fioIn.x/2 == fioOut.x &&
			fioIn.y/2 == fioOut.y );
	}

  int tile_size = 10;
  int cache_size = 2*(tile_size);
  int dimCache = cache_size*cache_size*cache_size*sizeof(float)*fioIn.iFeaturesPerVector;
  dim3 dimBlock(tile_size,tile_size,tile_size);
  dim3 dimGrid(ceil(fioOut.x/double(tile_size)), ceil(fioOut.y/double(tile_size)), ceil(fioOut.z/double(tile_size)));
  cudaSubSampleInterpolate<<<dimGrid, dimBlock, dimCache>>>(fioIn, fioOut, tile_size, cache_size);
  cudaMemcpy(fioOut.pfVectors, fioOut.d_pfVectors, iDataSizeFloat, cudaMemcpyDeviceToHost);
  return 1;

}

__global__ void cudaSubSampleInterpolate(
  FEATUREIO input,
  FEATUREIO output,
  int tile_size,
  int cache_size){

  extern __shared__ float shared_data[];

  int bOx = blockIdx.x * tile_size; // block x index for output
  int bOy = blockIdx.y * tile_size; // block x index for output
  int bOz = blockIdx.z * tile_size; // block x index for output
  int tOx = threadIdx.x; // thread x index for output
  int tOy = threadIdx.y; // thread y index for output
  int tOz = threadIdx.z; // thread z index for output
  int bIx = blockIdx.x * cache_size; // block x index for input
  int bIy = blockIdx.y * cache_size; // block y index for input
  int bIz = blockIdx.z * cache_size; // block z index for input

  int tile_id = tOz * tile_size * tile_size + tOy * tile_size + tOx; // id of thread for output

  // Only keep enought worker for cover the cache_size(xy) area (must use loop, don't enought thread to cover all the cache size)
  if (tile_id < cache_size * cache_size) {

    // Give a new 2D index to the thread : between (0,0) and (cache_size,cache_size)
    int tileZ = (int)((double)tile_id / (cache_size)) % (cache_size); // create a thread z index for input
    int tileY =  tile_id % (cache_size); // create a thread y index for input

    // Move the block pixel to left limit (3d view)
    int input_z = (bIz + tileZ);
    int input_y = (bIy + tileY);
    int input_x_root = bIx; // partial X =» seed for xAxis
    int input_x;
    // Make the xAxis Grow up
    for (int stemLength = 0; stemLength < cache_size; stemLength ++) {
      input_x = (input_x_root + stemLength);
      if (is_pixel(input_x, input_y, input_z, input.x, input.y, input.z)){
        shared_data[tileZ*cache_size*cache_size*input.iFeaturesPerVector + tileY*cache_size*input.iFeaturesPerVector + stemLength*input.iFeaturesPerVector] = input.d_pfVectors[(input_z)*input.x*input.y*input.iFeaturesPerVector + (input_y)*input.x*input.iFeaturesPerVector + input_x*input.iFeaturesPerVector];
      }
      else{
        shared_data[tileZ*cache_size*cache_size*input.iFeaturesPerVector + tileY*cache_size*input.iFeaturesPerVector + stemLength*input.iFeaturesPerVector] = 0.0;
      }
    }
  }
  __syncthreads();

  if (is_pixel(bOx+tOx, bOy+tOy, bOz+tOz, output.x, output.y, output.z)) {
    float data=shared_data[(tOx*2)*input.iFeaturesPerVector + (tOy*2)*cache_size*input.iFeaturesPerVector + (tOz*2)*cache_size*cache_size*input.iFeaturesPerVector];
    data+=shared_data[((tOx*2)+1)*input.iFeaturesPerVector + ((tOy*2))*cache_size*input.iFeaturesPerVector + ((tOz*2))*cache_size*cache_size*input.iFeaturesPerVector];
    data+=shared_data[((tOx*2)+1)*input.iFeaturesPerVector + ((tOy*2)+1)*cache_size*input.iFeaturesPerVector + ((tOz*2))*cache_size*cache_size*input.iFeaturesPerVector];
    data+=shared_data[((tOx*2)+1)*input.iFeaturesPerVector + ((tOy*2))*cache_size*input.iFeaturesPerVector + ((tOz*2)+1)*cache_size*cache_size*input.iFeaturesPerVector];
    data+=shared_data[((tOx*2)+1)*input.iFeaturesPerVector + ((tOy*2)+1)*cache_size*input.iFeaturesPerVector + ((tOz*2)+1)*cache_size*cache_size*input.iFeaturesPerVector];
    data+=shared_data[(tOx*2)*input.iFeaturesPerVector + ((tOy*2)+1)*cache_size*input.iFeaturesPerVector + ((tOz*2))*cache_size*cache_size*input.iFeaturesPerVector];
    data+=shared_data[(tOx*2)*input.iFeaturesPerVector + ((tOy*2)+1)*cache_size*input.iFeaturesPerVector + ((tOz*2)+1)*cache_size*cache_size*input.iFeaturesPerVector];
    data+=shared_data[(tOx*2)*input.iFeaturesPerVector + ((tOy*2))*cache_size*input.iFeaturesPerVector + ((tOz*2)+1)*cache_size*cache_size*input.iFeaturesPerVector];
    output.d_pfVectors[bOx+tOx*input.iFeaturesPerVector + (bOy+tOy)*output.x*input.iFeaturesPerVector + (bOz+tOz)*output.y*output.x*input.iFeaturesPerVector]=data*0.125;
  }
}

//
// fioCudaMultSum()
//
// Difference of Gaussian with cuda
//
__host__ int fioCudaMultSum(
    FEATUREIO &fioIn1,
		FEATUREIO &fioIn2,
		FEATUREIO &fioOut,
		const float &fMultIn2){
      if(
        fioIn1.x*fioIn1.y*fioIn1.z*fioIn1.iFeaturesPerVector != fioIn2.x*fioIn2.y*fioIn2.z*fioIn2.iFeaturesPerVector
        ||
        fioIn1.x*fioIn1.y*fioIn1.z*fioIn1.iFeaturesPerVector != fioOut.x*fioOut.y*fioOut.z*fioOut.iFeaturesPerVector
        ||
        fioIn2.x*fioIn2.y*fioIn2.z*fioIn2.iFeaturesPerVector != fioOut.x*fioOut.y*fioOut.z*fioOut.iFeaturesPerVector
        )
      {
        return 0;
      }
      // Define device parameters
      int iDataSizeFloat = fioIn1.x*fioIn1.y*fioIn1.z*fioIn1.iFeaturesPerVector*sizeof(float);
      int tile_size = 10;//kernel_size;
      dim3 dimGrid(ceil(fioIn1.x/double(tile_size)), ceil(fioIn1.y/double(tile_size)), ceil(fioIn1.z/double(tile_size)));
      dim3 dimBlock(tile_size,tile_size,tile_size);

      /*float *d_fioIn1; // INPUT IMAGE 1
      cudaMalloc((void**)&d_fioIn1, iDataSizeFloat); //alloc image
      float *array_h1=static_cast<float *>(fioIn1.pfVectors);//get a 1d array of pixel float
      cudaMemcpy(d_fioIn1, array_h1, iDataSizeFloat, cudaMemcpyHostToDevice); //get the array to device image

      float *d_fioIn2; // INPUT IMAGE 2
      cudaMalloc((void**)&d_fioIn2, iDataSizeFloat); //alloc image
      float *array_h2=static_cast<float *>(fioIn2.pfVectors);//get a 1d array of pixel float
      cudaMemcpy(d_fioIn2, array_h2, iDataSizeFloat, cudaMemcpyHostToDevice); //get the array to device image

      float *d_fioOut; // OUTPUT IMAGE
      cudaMalloc((float**)&d_fioOut, iDataSizeFloat); //alloc image
      cudaMemset(d_fioOut, 0, iDataSizeFloat); // Set all val to 0

      CudaMultSum<<<dimGrid, dimBlock>>>(d_fioIn1, d_fioIn2, d_fioOut, fioOut.x, fioOut.y, fioOut.z, fioOut.iFeaturesPerVector, fMultIn2);
      cudaDeviceSynchronize();

      cudaMemcpy(fioOut.pfVectors, d_fioOut, iDataSizeFloat, cudaMemcpyDeviceToHost);
      cudaFree(d_fioIn1);
      cudaFree(d_fioIn2);
      cudaFree(d_fioOut);*/

      cudaMemcpy(fioIn1.d_pfVectors, fioIn1.pfVectors, iDataSizeFloat, cudaMemcpyHostToDevice); //get the array to device image

      CudaMultSum<<<dimGrid, dimBlock>>>(fioIn1, fioIn2, fioOut, fMultIn2);
      cudaDeviceSynchronize();

      cudaMemcpy(fioOut.pfVectors, fioOut.d_pfVectors, iDataSizeFloat, cudaMemcpyDeviceToHost);
      return 1;
    }

__global__ void CudaMultSum(
    FEATUREIO input1,
    FEATUREIO input2,
    FEATUREIO output,
    float fMultIn2){

      int pos_X_b = blockIdx.x * blockDim.x + threadIdx.x;
      int pos_Y_b = blockIdx.y * blockDim.y + threadIdx.y;
      int pos_Z_b = blockIdx.z * blockDim.z + threadIdx.z;
      int index;
      int pos_X;
      int pos_Y;
      int pos_Z;

      pos_X = pos_X_b;
      pos_Y = pos_Y_b;
      pos_Z = pos_Z_b;
      index = pos_Z*input1.y*input1.x*input1.iFeaturesPerVector + pos_Y*input1.x*input1.iFeaturesPerVector+pos_X*input1.iFeaturesPerVector;
      if (is_pixel(pos_X, pos_Y, pos_Z, input1.x, input1.y, input1.z)) {
        for (int i = 0; i < input1.iFeaturesPerVector; i++) {
          output.d_pfVectors[index+i] = input1.d_pfVectors[index+i]+fMultIn2*input2.d_pfVectors[index+i];
        }
      }
          //}
        //}
      //}
    }
__device__ float sign(float x){
  return ((x > 0) - (x < 0));
}

__host__ void detectExtrema4D_test_cuda(
    FEATUREIO &inputH,
    FEATUREIO &inputC,
    FEATUREIO &fioSumOfSign,
    LOCATION_VALUE_XYZ_ARRAY &lvaMinima,
    LOCATION_VALUE_XYZ_ARRAY &lvaMaxima,
    int best_device_id
){
  cudaSetDevice(best_device_id);
  int iDataSizeFloat = inputC.x*inputC.y*inputC.z*inputC.iFeaturesPerVector*sizeof(float);
  int tile_size=10;
  dim3 dimGrid(ceil(inputC.x/double(tile_size)), ceil(inputC.y/double(tile_size)), ceil(inputC.z/double(tile_size)));
  dim3 dimBlock(tile_size,tile_size,tile_size);
  int cache_size=tile_size + (1 * 2);
  int dimCache = (cache_size*cache_size*cache_size*4)*2;
  d_detectExtrema4D_test<<<dimGrid, dimBlock, dimCache>>>(inputC, inputH, fioSumOfSign, tile_size, cache_size);
  cudaMemcpy(fioSumOfSign.pfVectors, fioSumOfSign.d_pfVectors, iDataSizeFloat, cudaMemcpyDeviceToHost);
  lvaMaxima.iCount=0;
	lvaMinima.iCount=0;
  for(int z=1; z<fioSumOfSign.z-1; z++){
    for(int y=1; y<fioSumOfSign.y-1; y++){
      for(int x=1; x<fioSumOfSign.x-1; x++){
        int iIndex = x + y*fioSumOfSign.x + z*fioSumOfSign.y*fioSumOfSign.x;
        //printf("%f\n", fioSumOfSign.pfVectors[iIndex]);
        if (fioSumOfSign.pfVectors[iIndex]==53) {
          lvaMinima.plvz[lvaMinima.iCount].x = x;
          lvaMinima.plvz[lvaMinima.iCount].y = y;
          lvaMinima.plvz[lvaMinima.iCount].z = z;
          lvaMinima.plvz[lvaMinima.iCount].fValue = inputC.pfVectors[iIndex];
          lvaMinima.iCount++;
        }
        else if(fioSumOfSign.pfVectors[iIndex]==-53){
          lvaMaxima.plvz[lvaMaxima.iCount].x = x;
          lvaMaxima.plvz[lvaMaxima.iCount].y = y;
          lvaMaxima.plvz[lvaMaxima.iCount].z = z;
          lvaMaxima.plvz[lvaMaxima.iCount].fValue = inputC.pfVectors[iIndex];
          lvaMaxima.iCount++;
        }
        else{
          continue;
        }
      }
    }
  }
}

__global__ void d_detectExtrema4D_test(
    FEATUREIO inputC,
    FEATUREIO inputH,
    FEATUREIO output,
    int tile_size,
    int cache_size
){

  extern __shared__ float shared_data[];

  int bx = blockIdx.x * tile_size;
  int by = blockIdx.y * tile_size;
  int bz = blockIdx.z * tile_size;

  // Set seed pixels coordinates
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  // Generate xyz position for center pixel
  int x_pos = bx + tx;
  int y_pos = by + ty;
  int z_pos = bz + tz;
  int tile_id = tz * tile_size * tile_size + ty * tile_size + tx; // before kernel size


  // Only keep enought worker for cover the cache_size(xy) area
  if (tile_id < cache_size * cache_size) {
    int tileZ = (int)((double)tile_id / (cache_size)) % (cache_size);
    int tileY =  tile_id % (cache_size);

    // Move the block pixel to upper left limit (3d view)
    int input_z = bz + tileZ-1;
    int input_y = by + tileY-1;
    int input_x_root = bx-1; // partial Z =» seed for zAxis
    int input_x;

    // Make the zAxis Grow up
    for (int stemLength = 0; stemLength < cache_size; stemLength ++) {
      input_x = input_x_root + stemLength;
      if (is_pixel(input_x, input_y, input_z, inputC.x, inputC.y, inputC.z)){
        shared_data[tileZ*cache_size*cache_size + tileY*cache_size + stemLength] = inputC.d_pfVectors[input_z*inputC.y*inputC.x + input_y*inputC.x + input_x];
        shared_data[(cache_size*cache_size*cache_size) + tileZ*cache_size*cache_size + tileY*cache_size + stemLength] = inputH.d_pfVectors[input_z*inputC.y*inputC.x + input_y*inputC.x + input_x];
      } else {
        shared_data[tileZ*cache_size*cache_size + tileY*cache_size + stemLength] = 0.0f;
        shared_data[(cache_size*cache_size*cache_size) + tileZ*cache_size*cache_size + tileY*cache_size + stemLength] = 0.0f;
      }
    }
  }
    // Wait for other thread to complete the job (and fill shared_data)
    __syncthreads();

    if (is_pixel(x_pos, y_pos, z_pos, inputC.x, inputC.y, inputC.z)) {
      float xOutputValue = 0.0;
      for (int z = 0; z < 3; z ++) {
        for (int y = 0; y < 3; y ++) {
          for (int x = 0; x < 3; x ++) {
              xOutputValue +=
                sign(shared_data[(tx + x) + (ty+y)*(cache_size) + (tz+z)*(cache_size)*(cache_size)] - shared_data[(tx+1) + (ty+1)*(cache_size) + (tz+1)*(cache_size)*(cache_size)]);
              xOutputValue +=
                sign(shared_data[(cache_size*cache_size*cache_size) + (tx + x) + (ty+y)*(cache_size) + (tz+z)*(cache_size)*(cache_size)] - shared_data[(tx+1) + (ty+1)*(cache_size) + (tz+1)*(cache_size)*(cache_size)]);
            }
          }
        }
      output.d_pfVectors[(z_pos)*(inputC.x)*(inputC.y) + (y_pos)*(inputC.x) + x_pos] = (xOutputValue);
    }
}
