/*
CUDA - generation of array of N elements and calculates even and odd numbers occurence - no streams
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define DEBUG 0
#define MAX_NUM 1000 
__host__
void errorexit(const char *s) {
		printf("\n%s\n",s); 
		exit(EXIT_FAILURE);   
}

__host__ 
void generate(int *matrix, int matrixSize) {
	srand(time(NULL));
	for(int i=0; i<matrixSize; i++) {
		matrix[i] = rand()%MAX_NUM;
	}
}

__global__ 
void calculation(int *matrix, int *histogram, int matrixSize) {
		int my_index=blockIdx.x*blockDim.x+threadIdx.x;
		if (my_index < matrixSize) {
			atomicAdd(&(histogram[matrix[my_index]]), 1);
		} 
}

int main(int argc,char **argv) {

	//define array size and allocate memory on host
	int matrixSize=100000;
	int *hMatrix=(int*)malloc(matrixSize*sizeof(int));
	
	//generate random numbers
	generate(hMatrix, matrixSize);

	if(DEBUG) {
		printf("Generated numbers: \n");
		for(int i=0; i<matrixSize; i++) {
			printf("%d ", hMatrix[i]);
		}
		printf("\n");
	}

	//allocate memory for odd and even numbers counters - host
	int *hHistogram=(int*)malloc((MAX_NUM + 1) * sizeof(int));

	//allocate memory for odd and even numbers counters and array - device
	int *dHistogram = NULL;
	int *dMatrix=NULL;

	if (cudaSuccess!=cudaMalloc((void **)&dHistogram, (MAX_NUM + 1) * sizeof(int)))
			errorexit("Error allocating memory on the GPU");

	if (cudaSuccess!=cudaMalloc((void **)&dMatrix,matrixSize*sizeof(int)))
			errorexit("Error allocating memory on the GPU");

	//initialize allocated counters with 0
	if (cudaSuccess!=cudaMemset(dHistogram,0, (MAX_NUM + 1) * sizeof(int)))
			errorexit("Error initializing memory on the GPU");

	//copy array to device
	if (cudaSuccess!=cudaMemcpy(dMatrix,hMatrix,matrixSize*sizeof(int),cudaMemcpyHostToDevice))
		 errorexit("Error copying input data to device");

	int threadsinblock=1024;
	int blocksingrid=1+((matrixSize-1)/threadsinblock); 

	//run kernel on GPU 
	calculation<<<blocksingrid, threadsinblock>>>(dMatrix, dHistogram, matrixSize);

	//copy results from GPU
	if (cudaSuccess!=cudaMemcpy(hHistogram, dHistogram, (MAX_NUM + 1) * sizeof(int),cudaMemcpyDeviceToHost))
		 errorexit("Error copying results");

	int totalNumber = 0;	 

	for(int i=0; i<MAX_NUM; i++) {
		printf("number %d : %d \n", i , hHistogram[i]);
		totalNumber += hHistogram[i];
	}

	printf("Totlan numbers count in the histogram is %d\n", i , totalNumber);

	//Free memory
	free(hHistogram);
	free(hMatrix);
		
	if (cudaSuccess!=cudaFree(dHistogram))
		errorexit("Error when deallocating space on the GPU");
	if (cudaSuccess!=cudaFree(dMatrix))
		errorexit("Error when deallocating space on the GPU");
}
