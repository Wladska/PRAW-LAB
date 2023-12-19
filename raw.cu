/*
CUDA - generation and sum of arithmetic progression build of 10240000 elements a1=0 r=1 without  Unified Memory
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__host__
void errorexit(const char *s) {
    printf("\n%s",s);	
    exit(EXIT_FAILURE);	 	
}

//elements generation
__global__ 
void calculate(int *result, unsigned long long NUMBER, unsigned long long BIGGEST_NUMBER_TO_CHECK) {
    int my_index=blockIdx.x*blockDim.x+threadIdx.x;
    if (my_index >= 2 && my_index <= BIGGEST_NUMBER_TO_CHECK && NUMBER % my_index == 0) {
        atomicAdd(&(result[blockIdx.x]), 1);
    }
}


int main(int argc,char **argv) {
    unsigned long long NUMBER  = 7830817777;
    unsigned long long BIGGEST_NUMBER_TO_CHECK  = sqrt(7830817777);
    int threadsinblock = 1024;
    int blocksingrid= BIGGEST_NUMBER_TO_CHECK / threadsinblock;	

    int *host_results = (int*)malloc(blocksingrid * sizeof(int));
    int *results=NULL;

    //unified memory allocation - available for host and device
    if (cudaSuccess!=cudaMalloc(&results,blocksingrid * sizeof(int)))
      errorexit("Error allocating memory on the GPU");

    //initialize allocated counters with 0
    if (cudaSuccess!=cudaMemset(results, 0, blocksingrid * sizeof(int)))
      errorexit("Error initializing memory on the GPU");

    //call to GPU - kernel execution 
    calculate<<<blocksingrid,threadsinblock>>>(results, NUMBER, BIGGEST_NUMBER_TO_CHECK);

    if (cudaSuccess!=cudaGetLastError())
      errorexit("Error during kernel launch");
  
    if (cudaSuccess!=cudaMemcpy(host_results, results, blocksingrid * sizeof(int), cudaMemcpyDeviceToHost))
       errorexit("Error copying results");

    //calculate sum of all elements
    int result = 0;
    for(int i=0; i<blocksingrid; i++) {
      result += results[i];
    }

    if (result == 0) {
      printf("Number IS prime\n");
    } else {
      printf("Number IS NOT prime\n");
    }

    //free memory
    if (cudaSuccess!=cudaFree(results))
      errorexit("Error when deallocating space on the GPU");
}
