
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>

__global__  void vector_sub(double *out, const double *a, double c, int m, int n) {
    int tid =threadIdx.x;

    if (tid >= m && tid < n) {
        out[tid] -= a[tid] * c;
    }
}

void printMatrix(double **matrix, const int *SIZE) {
    for (int i = 0; i < *SIZE; ++i) {
        for (int j = 0; j < *SIZE; ++j) {
            printf("%f ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

double diagonalMultiplication(double **matrix, const int *SIZE) {
    double rez = 1;
    for (int i = 0; i < *SIZE; ++i) rez *= matrix[i][i];
    return rez;
}

int zeroesCheck(const double *range, const int *SIZE) {
    int count = 0, flag = 1;
    for (int i = 0; i < *SIZE; ++i)
        if (range[i] == 0 && flag) {
            count++;
        } else flag = 0;
    return count;
}

int power(int a, int b) {
    int rez = 1;
    for (int i = 0; i < b; ++i) rez *= a;
    return rez;
}

int sort(double **matrix, int *SIZE) {
    int i, j, count = 0;
    double *temp;
    for (i = 0; i < *SIZE - 1; i++)
        for (j = 0; j < *SIZE - i - 1; j++) {
            if (zeroesCheck(matrix[j], SIZE) > zeroesCheck(matrix[j + 1], SIZE)) {
                count++;
                temp = matrix[j];
                matrix[j] = matrix[j + 1];
                matrix[j + 1] = temp;
            }
        }
    return power(-1, count);
}


double gaussianDeterminant(double **matrix, int* SIZE) {
    int size = *SIZE;
    double first, factor;
    double *d_a, *d_out;

    cudaMalloc((void **) &d_a, sizeof(double) * *SIZE);
    cudaMalloc((void **) &d_out, sizeof(double) * *SIZE);

    while (size > 1) {
        if (matrix[*SIZE - size][*SIZE - size] == 0) return 0;
        first = matrix[*SIZE - size][*SIZE - size];
        for (int i = *SIZE - size + 1; i < *SIZE; ++i) {

            factor = matrix[i][*SIZE - size] / first;
            cudaMemcpy(d_out, matrix[i], sizeof(double) * *SIZE, cudaMemcpyHostToDevice);
            cudaMemcpy(d_a, matrix[*SIZE - size], sizeof(double) * *SIZE, cudaMemcpyHostToDevice);

            vector_sub <<< 1, *SIZE >>>(d_out, d_a, factor, *SIZE - size, *SIZE);
            cudaMemcpy(matrix[i], d_out, sizeof(double) * *SIZE, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
        }

        size--;
    }
    cudaFree(d_a);
    cudaFree(d_out);
    return diagonalMultiplication(matrix, SIZE);
}

void init() {
    FILE *fp1, *fp2;
    if ((fp1 = fopen("read.txt", "r")) == nullptr) {
        printf("Can't open file 'read.txt'\n");
        exit(-1);
    }
    if ((fp2 = fopen("write.txt", "w")) == nullptr) {
        printf("Can't open file 'write.txt'\n");
        exit(-1);
    }
    double **matrix;
    double determinant;
    int SIZE, sign;
    clock_t time_start, time_finish;

    while (fscanf(fp1, "%d", &SIZE) == 1) {
        matrix = (double **) malloc(SIZE * sizeof(double *));
        for (int i = 0; i < SIZE; ++i) {
            matrix[i] = (double *) malloc(SIZE * sizeof(double));
            for (int j = 0; j < SIZE; ++j) {
                fscanf(fp1, "%lf", &matrix[i][j]);
            }
        }
        time_start = clock();
        sign = sort(matrix, &SIZE);
        determinant = gaussianDeterminant(matrix, &SIZE) * (double) sign;
        time_finish = clock();
        fprintf(fp2, "%ld %f\n", time_finish - time_start, determinant);
        for (int i = 0; i < SIZE; ++i) free(matrix[i]);
        free(matrix);
        if (determinant > DBL_MAX) exit(-2);
    }
    fclose(fp1);
    fclose(fp2);
}

int main() {
    init();
    return 0;
}




