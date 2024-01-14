#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <ctime>
#include <random>
#include <cmath>
#include <iostream>

#define ARRAY_SIZE 100

#define MIN_DISTRIBUTION -10000
#define MAX_DISTRIBUTION 10000

void swap(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
}

int partition(int* array, int low, int high) {
    int pivot = array[high];
    int i = low - 1;

    #pragma omp parallel for
    for (int j = low; j <= high; j++) {
        if (array[j] < pivot) {
            i++;
            swap(array[i], array[j]);
        }
    }
    swap(array[i + 1], array[high]);
    return i + 1;
}

void quicksort(int* array, int low, int high) {
  if (low < high) {
    int pivotIdx = partition(array, low, high);
    #pragma omp parallel
    {
      #pragma omp task
      {
          quicksort(array, low, pivotIdx - 1);
      }

      #pragma omp task
      {
          quicksort(array, pivotIdx + 1, high);
      }
    }
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
  int* randomNumbers = generateRandomInput();

  std::cout<< "Input: " << std::endl;
  // Print the input array
  for (int i = 0; i < ARRAY_SIZE; i++) {
    std::cout << randomNumbers[i] << " ";
  }
  std::cout << std::endl;

  quicksort(randomNumbers, 0, ARRAY_SIZE-1);

  std::cout<< "Sorted: " << std::endl;
  // Print the sorted array
  for (int i = 0; i < ARRAY_SIZE; i++) {
    std::cout << randomNumbers[i] << " ";
  }
  std::cout << std::endl;

  delete[] randomNumbers;
}