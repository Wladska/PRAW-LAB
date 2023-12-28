#include <stdio.h>
#include <stdlib.h>
#include<math.h>
#include <omp.h>

#define BUFFER_SIZE (512 * 1024) // 512kB buffer size

long getFileSize(const char* fileName){
    FILE* file = fopen(fileName, "rb");
    if (!file) {
        fprintf(stderr, "Error opening file.\n");
        return 0L;
    }
    
    fseek(file, 0, SEEK_END);
    long fileSize = ftell(file);
    rewind(file);
    
    fclose(file);
    return fileSize;
}

void reverseFile(const char* inputFileName, const char* outputFileName) {
    long inputFileSize = getFileSize(inputFileName);

    FILE* inputFile = fopen(inputFileName, "rb");
    if (!inputFile) {
        fprintf(stderr, "Error opening input file.\n");
        return;
    }

    FILE* outputFile = fopen(outputFileName, "wb");
    if (!outputFile) {
        fprintf(stderr, "Error opening output file.\n");
        fclose(inputFile);
        return;
    }

    // Calculate the number of chunks and number of threads based on buffer size
    int numThreads = inputFileSize / BUFFER_SIZE + 1;

    // Use OpenMP to parallelize the reverse operation
    #pragma omp parallel num_threads(numThreads)
    {
        int threadID = omp_get_thread_num();

        // Calculate the starting position for this thread
        long start = threadID * BUFFER_SIZE;

        // Calculate the actual size to read for this thread
        long size = start + BUFFER_SIZE > inputFileSize ? inputFileSize - start : BUFFER_SIZE;

        // Allocate buffer for this thread
        char* buffer = (char*)malloc(size);
        if (!buffer) {
            fprintf(stderr, "Memory allocation failed for thread %d.\n", threadID);
            return; // Skip this thread if memory allocation fails
        }

        // Read data into the buffer
        fseek(inputFile, start, SEEK_SET);
        fread(buffer, 1, size, inputFile);

        // Reverse the buffer
        for (long i = 0, j = size - 1; i < j; ++i, --j) {
            char temp = buffer[i];
            buffer[i] = buffer[j];
            buffer[j] = temp;
        }

        // Calculate the position to write to in the output file
        long writePosition = inputFileSize - (start + size);

        // Write the reversed buffer to the output file
        fseek(outputFile, writePosition, SEEK_SET);
        fwrite(buffer, 1, size, outputFile);

        // Free allocated buffer
        free(buffer);
    }

    // Close files
    fclose(inputFile);
    fclose(outputFile);
}

int main() {
    const char* inputFileName = "input.txt";
    const char* outputFileName = "output.txt";

    reverseFile(inputFileName, outputFileName);

    printf("File reversed successfully!\n");

    return 0;
}