/*
CUDA - dynamic parallelism sample
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <ctime>
#include <random>
#include <cmath>

#define ARRAY_SIZE 100

#define MIN_DISTRIBUTION -10000
#define MAX_DISTRIBUTION 10000

#define N 20

__host__
void errorexit(const char *s) {
		printf("\n%s\n",s); 
		exit(EXIT_FAILURE);   
}

__device__ void swap(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
}

__device__ int partition(int* array, int low, int high) {
    int pivot = array[high];
    int i = low - 1;

    for (int j = low; j <= high; j++) {
        if (array[j] < pivot) {
            i++;
            swap(array[i], array[j]);
        }
    }
    swap(array[i + 1], array[high]);
    return i + 1;
}

__global__ 
void quicksort(int* array, int low, int high) {
    if (low < high) {
        int pivotIdx = partition(array, low, high);

        quicksort<<<1, 1>>>(array, low, pivotIdx - 1);
        quicksort<<<1, 1>>>(array, pivotIdx + 1, high);
    }
}

int* generateRandomInput() {
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    std::default_random_engine generator(std::rand());
    std::uniform_int_distribution<int> distribution(MIN_DISTRIBUTION, MAX_DISTRIBUTION);

    int* randomNumbers = new int[ARRAY_SIZE];

    for (int i = 0; i < ARRAY_SIZE; ++i) {
        randomNumbers[i] = distribution(generator);
    }

    return randomNumbers;
}

int main(int argc,char **argv) {
	//run kernel on GPU 
    int* randomNumbers = generateRandomInput();
	
    std::cout<< "Input: " << std::endl;
    // Print the input array
    for (int i = 0; i < ARRAY_SIZE; i++) {
        std::cout << randomNumbers[i] << " ";
    }
    std::cout << std::endl;

    int *inputData;
    cudaMalloc(&inputData, ARRAY_SIZE * sizeof(int));
    cudaMemcpy(inputData, randomNumbers, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);

	quicksort<<<1, 1>>>(inputData, 0, ARRAY_SIZE-1);
    cudaDeviceSynchronize();

    cudaMemcpy(randomNumbers, inputData, ARRAY_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(inputData);

	if (cudaSuccess!=cudaGetLastError())
      errorexit("Error during kernel launch");

    std::cout<< "Sorted: " << std::endl;
    // Print the sorted array
    for (int i = 0; i < ARRAY_SIZE; i++) {
        std::cout << randomNumbers[i] << " ";
    }
    std::cout << std::endl;

    delete[] randomNumbers;
}
