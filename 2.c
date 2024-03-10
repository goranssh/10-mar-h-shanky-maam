#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <mpi.h>

#define MAX_MATRIX_SIZE 10

void generateRandomMatrix(int size, int matrix[size][size]) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i][j] = rand() % 10; // Generate random values between 0 and 9
        }
    }
}

void multiplyMatrices(int size, int matrixA[size][size], int matrixB[size][size], int result[size][size]) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            result[i][j] = 0;
            for (int k = 0; k < size; k++) {
                result[i][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    }
}

void writeToFile(int size, int matrix[size][size], const char *filename) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error opening file.\n");
        return;
    }
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            fprintf(file, "%d ", matrix[i][j]);
        }
        fprintf(file, "\n");
    }
    fclose(file);
}

int main(int argc, char **argv) {
    srand(time(NULL)); // Seed the random number generator

    MPI_Init(&argc, &argv);

    int processRank, totalProcesses;
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
    MPI_Comm_size(MPI_COMM_WORLD, &totalProcesses);

    int matrixSize = rand() % MAX_MATRIX_SIZE + 1; // Random matrix size between 1 and 10
    printf("Matrix Size: %d x %d\n", matrixSize, matrixSize);
    int matrixA[matrixSize][matrixSize], matrixB[matrixSize][matrixSize], resultMatrix[matrixSize][matrixSize];

    if (processRank == 0) {
        // Generate random matrices A and B
        generateRandomMatrix(matrixSize, matrixA);
        generateRandomMatrix(matrixSize, matrixB);
    }

    // Broadcast matrices A and B to all processes
    MPI_Bcast(matrixA, matrixSize * matrixSize, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(matrixB, matrixSize * matrixSize, MPI_INT, 0, MPI_COMM_WORLD);

    struct timespec startTime, endTime;
    clock_gettime(CLOCK_MONOTONIC_RAW, &startTime);

    // Divide work among processes
    int chunkSize = matrixSize / totalProcesses;
    int startRow = processRank * chunkSize;
    int endRow = (processRank == totalProcesses - 1) ? matrixSize : (processRank + 1) * chunkSize;

    // Multiply matrices A and B
    multiplyMatrices(matrixSize, matrixA, matrixB, resultMatrix);

    // Gather results from all processes to process 0
    MPI_Gather(resultMatrix + startRow, chunkSize * matrixSize, MPI_INT, resultMatrix, chunkSize * matrixSize, MPI_INT, 0, MPI_COMM_WORLD);

    clock_gettime(CLOCK_MONOTONIC_RAW, &endTime);
    double executionTime = (endTime.tv_sec - startTime.tv_sec) * 1e6 + (endTime.tv_nsec - startTime.tv_nsec) / 1e3;

    // Print execution time on process 0
    if (processRank == 0) {
        printf("Execution time: %.6f microseconds\n", executionTime);
    }

    MPI_Finalize();
    return 0;
}
