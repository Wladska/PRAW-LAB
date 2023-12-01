/*
CUDA - generation and sum of arithmetic progression build of 10240000 elements a1=0 r=1 with shared memory
*/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>

__host__
void errorexit(const char *s) {
    printf("\n%s",s); 
    exit(EXIT_FAILURE);   
}

//elements generation
__global__ 
void calculate(float *result) {
    //allocate memory - size same as number of threads in block
   __shared__ float sresults[1024];  
    int my_index=blockIdx.x*blockDim.x+threadIdx.x;

    sresults[threadIdx.x]= 1 / powf(2, my_index);
    __syncthreads();

    //calculate sum of all elements of thread within same block using shared memory 
   for (unsigned int s = 1; s < blockDim .x; s *= 2) {
      if ( threadIdx.x % (2 * s) == 0) {
        sresults[ threadIdx.x ] += sresults[ threadIdx.x + s ];
      }
      __syncthreads ();   
    }

    //first thread in block write results of this block to global memory
    if (threadIdx.x==0) {
      result[blockIdx.x]=sresults[0];
    }
}

int main(int argc,char **argv) {

    float result;
    int threadsinblock=1024;
    int blocksingrid=10000;	

    int size = threadsinblock*blocksingrid;
    //memory allocation on host
    float *hresults=(float*)malloc(size*sizeof(float));
    if (!hresults) errorexit("Error allocating memory on the host");	

    float *dresults=NULL;
    //devie memory allocation (GPU)
    if (cudaSuccess!=cudaMalloc((void **)&dresults,size*sizeof(float)))
      errorexit("Error allocating memory on the GPU");

    //call to GPU - kernel execution 
    calculate<<<blocksingrid,threadsinblock>>>(dresults);
    if (cudaSuccess!=cudaGetLastError())
      errorexit("Error during kernel launch");
  
    //getting results from GPU to host memory
    if (cudaSuccess!=cudaMemcpy(hresults,dresults,size*sizeof(float),cudaMemcpyDeviceToHost))
       errorexit("Error copying results");


    //calculate sum of all elements
    result=0;

    for(int i=0; i<blocksingrid; i++) {
      result = result + hresults[i * threadsinblock];
    }

    std::cout << "\nThe final result is " << result << std::endl;

    //free memory
    free(hresults);
    if (cudaSuccess!=cudaFree(dresults))
      errorexit("Error when deallocating space on the GPU");

}

