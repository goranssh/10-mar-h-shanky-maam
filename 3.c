#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <CL/cl.h>
#include <mpi.h>

#define MAX_MATRIX_SIZE 90
#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
        printf("Error: %d\n", err); \
        return EXIT_FAILURE; \
    }

void generateRandomMatrix(int size, int *matrix) {
    for (int i = 0; i < size * size; i++) {
        matrix[i] = rand() % 10; // Generate random values between 0 and 9
    }
}

void printMatrix(int size, int *matrix) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%d ", matrix[i * size + j]);
        }
        printf("\n");
    }
}

void writeToFile(int size, int *matrix, const char *filename) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error opening file.\n");
        return;
    }
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            fprintf(file, "%d ", matrix[i * size + j]);
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

    int matrixSize = rand() % MAX_MATRIX_SIZE + 1; // Random matrix size between 1 and 90
    printf("Matrix Size: %d x %d\n", matrixSize, matrixSize);

    int *matrixA = NULL, *matrixB = NULL, *matrixC = NULL;

    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_int err;

    // Initialize OpenCL
    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_ERROR(err);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_ERROR(err);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_ERROR(err);
    queue = clCreateCommandQueue(context, device, 0, &err);
    CHECK_ERROR(err);

    // Load kernel source code
    FILE *file = fopen("matrix_multiply.cl", "r");
    if (!file) {
        printf("Failed to open kernel file.\n");
        return EXIT_FAILURE;
    }
    fseek(file, 0, SEEK_END);
    size_t kernelSourceSize = ftell(file);
    rewind(file);
    char *kernelSource = (char *)malloc(kernelSourceSize + 1);
    fread(kernelSource, 1, kernelSourceSize, file);
    kernelSource[kernelSourceSize] = '\0';
    fclose(file);

    // Create program
    program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, &kernelSourceSize, &err);
    CHECK_ERROR(err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error building program.\n");
        char buildLog[4096];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);
        printf("Build log:\n%s\n", buildLog);
        return EXIT_FAILURE;
    }

    // Create kernel
    kernel = clCreateKernel(program, "matrix_multiply", &err);
    CHECK_ERROR(err);

    if (processRank == 0) {
        // Generate random matrices A and B
        matrixA = (int *)malloc(matrixSize * matrixSize * sizeof(int));
        matrixB = (int *)malloc(matrixSize * matrixSize * sizeof(int));
        generateRandomMatrix(matrixSize, matrixA);
        generateRandomMatrix(matrixSize, matrixB);
    }

    // Broadcast matrices A and B to all processes
    MPI_Bcast(matrixA, matrixSize * matrixSize, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(matrixB, matrixSize * matrixSize, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate memory for result matrix C in each process
    int chunkSize = matrixSize / totalProcesses;
    int startRow = processRank * chunkSize;
    int endRow = (processRank == totalProcesses - 1) ? matrixSize : (processRank + 1) * chunkSize;
    matrixC = (int *)malloc(chunkSize * matrixSize * sizeof(int));

    // Create buffer objects for matrices A, B, and C
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, matrixSize * matrixSize * sizeof(int), matrixA, &err);
    CHECK_ERROR(err);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, matrixSize * matrixSize * sizeof(int), matrixB, &err);
    CHECK_ERROR(err);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, chunkSize * matrixSize * sizeof(int), NULL, &err);
    CHECK_ERROR(err);

    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 3, sizeof(cl_int), &matrixSize);
    CHECK_ERROR(err);

    // Execute kernel
    size_t globalWorkSize[2] = {matrixSize, chunkSize};
    size_t localWorkSize[2] = {1, 1};
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Read result matrix C from device to host
    err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, chunkSize * matrixSize * sizeof(int), matrixC, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Gather results from all processes to process 0
    MPI_Gather(matrixC, chunkSize * matrixSize, MPI_INT, matrixC, chunkSize * matrixSize, MPI_INT, 0, MPI_COMM_WORLD);

    // Print matrix C on process 0
    if (processRank == 0) {
        printf("Resultant Matrix C:\n");
        printMatrix(matrixSize, matrixC);

        // Write matrix C to a file
        writeToFile(matrixSize, matrixC, "result.txt");
    }

    // Cleanup
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(kernelSource);
    free(matrixA);
    free(matrixB);
    free(matrixC);

    MPI_Finalize();

    return 0;
}
