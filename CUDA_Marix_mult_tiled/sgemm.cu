#include <stdio.h>
#include <chrono>
#include<assert.h>
#include <stdlib.h>
#include <string>

__global__ void matrixMulKernel_1thread1elemen(float* A_d, float* B_d, float* C_d, int N, int M, int K){
    unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;
    if(row < M && col < N){
        float sum = 0.0f;
        for(unsigned int i = 0; i <K; ++i){
            sum +=A_d[row*K + i] * B_d[i*K + col];
        }
        C_d[row*K + col] = sum;
    }
}

__global__ void matrixMulKernel_tiled(float* A_d, float* B_d, float* C_d, int N, int M, int K){
    extern __shared__ float Total[];
    float* A_s = &Total[0];
    float* B_s = &Total[max(2,K/2)*2];

    unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

    float sum = 0.0f;
    for(unsigned int i = 0; i < K/(max(2,K/2)*2); ++i){
        A_s[threadIdx.y*blockDim.x+threadIdx.x] = A_d[row*N + i *(max(2,K/2)*2)+threadIdx.x];
        B_s[threadIdx.y*blockDim.x+threadIdx.x] = B_d[(i*(max(2,K/2)*2) + threadIdx.y)*K + col];
        __syncthreads();

        for(unsigned int j = 0; j < (max(2,K/2)*2); ++i){
            sum += A_s[threadIdx.y*blockDim.x+j]*B_s[j*blockDim.x+threadIdx.x];
        }
        __syncthreads();
    }

    C_d[row*K + col] =sum;
}

bool verify_result(float *a, float *b, float *c, float *v, int n, int m){
    for (int i = 0; i < n*m; i++) {
    if(c[i] != v[i]){
        return false;
    }
  }

  return true;
}

void basicSgemm_d_1thread1element (int m, int k, int n, const float *A_h, const float*B_h, float* C_h){
    float *A_d, *B_d, *C_d;
    auto start_time = std::chrono::high_resolution_clock::now();
    cudaMalloc(&A_d,sizeof(float)*(m*k));
    cudaMalloc(&B_d,sizeof(float)*(k*n));
    cudaMalloc(&C_d,sizeof(float)*(m*n));
    auto end_time = std::chrono::high_resolution_clock::now();
    float time_to_add_seconds = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1e6f;

    printf("    cudaMalloc: %24s%9.6fs \n", "",time_to_add_seconds);

    start_time = std::chrono::high_resolution_clock::now();
    cudaMemcpy(A_d,A_h,sizeof(float)*(m*k),cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(B_d,B_h,sizeof(float)*(k*n),cudaMemcpyKind::cudaMemcpyHostToDevice);
    end_time = std::chrono::high_resolution_clock::now();
    time_to_add_seconds = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1e6f;
    
    printf("    cudaMemcpy: %24s%9.6fs \n", "", time_to_add_seconds);

    start_time = std::chrono::high_resolution_clock::now();
    int dimSize = 0;
    if(m > n){
        dimSize = std::ceil(m/1024.0f);
    }else{
        dimSize = std::ceil(n/1024.0f);
    }
    dim3 blockDim(dimSize,dimSize,1);
    dim3 gridDim(m,n,1);
    matrixMulKernel_1thread1elemen<<<gridDim,blockDim>>>(A_d,B_d,C_d,n,m,k);
    cudaDeviceSynchronize();
    end_time = std::chrono::high_resolution_clock::now();
    time_to_add_seconds = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1e6f;
    printf("    matrixMulKernel_1thread1elemen<<<(%d,%d,%d),(%d,%d,%d)>>>: %9.6f \n",dimSize,dimSize,1,m,n,1,  time_to_add_seconds);
    cudaDeviceSynchronize();

    start_time = std::chrono::high_resolution_clock::now();
    cudaMemcpy(C_h,C_d,sizeof(float)*(m*n),cudaMemcpyKind::cudaMemcpyDeviceToHost);
    end_time = std::chrono::high_resolution_clock::now();
    time_to_add_seconds = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1e6f;
    printf("    CudaMemcpy: %24s%9.6fs \n", "", time_to_add_seconds);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

void basicSgemm_tiled (int m, int k, int n, const float *A_h, const float*B_h, float* C_h ,float* V_h){
    float *A_d, *B_d, *C_d;
    auto start_time = std::chrono::high_resolution_clock::now();
    cudaMalloc(&A_d,sizeof(float)*(m*k));
    cudaMalloc(&B_d,sizeof(float)*(k*n));
    cudaMalloc(&C_d,sizeof(float)*(m*n));
    auto end_time = std::chrono::high_resolution_clock::now();
    float time_to_add_seconds = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1e6f;

    printf("    cudaMalloc: %24s%9.6fs \n", "",time_to_add_seconds);

    start_time = std::chrono::high_resolution_clock::now();
    cudaMemcpy(A_d,A_h,sizeof(float)*(m*k),cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(B_d,B_h,sizeof(float)*(k*n),cudaMemcpyKind::cudaMemcpyHostToDevice);
    end_time = std::chrono::high_resolution_clock::now();
    time_to_add_seconds = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1e6f;
    
    printf("    cudaMemcpy: %24s%9.6fs \n", "", time_to_add_seconds);

    start_time = std::chrono::high_resolution_clock::now();
    int dimSize = 0;
    if(m > n){
        dimSize = std::ceil(m/1024.0f);
    }else{
        dimSize = std::ceil(n/1024.0f);
    }
    dim3 blockDim(dimSize,dimSize,1);
    dim3 gridDim(m,n,1);
    matrixMulKernel_tiled<<<gridDim,blockDim,max(2,k/2)*4>>>(A_d,B_d,C_d,n,m,k);
    cudaDeviceSynchronize();
    end_time = std::chrono::high_resolution_clock::now();
    time_to_add_seconds = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1e6f;
    printf("    matrixMulKernel_tiled<<<(%d,%d,%d),(%d,%d,%d)>>>: %9.6f \n",dimSize,dimSize,1,m,n,1,  time_to_add_seconds);
    cudaDeviceSynchronize();

    start_time = std::chrono::high_resolution_clock::now();
    cudaMemcpy(C_h,C_d,sizeof(float)*(m*n),cudaMemcpyKind::cudaMemcpyDeviceToHost);
    end_time = std::chrono::high_resolution_clock::now();
    time_to_add_seconds = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1e6f;
    printf("    CudaMemcpy: %24s%9.6fs \n", "", time_to_add_seconds);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main(int argc, char *argv[])
{
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int m = std::stoi(argv[1]);
    int k = std::stoi(argv[2]);
    int n = std::stoi(argv[3]);
    float* A_h = (float*)malloc(sizeof(float)*(m*k));
    float* B_h = (float*)malloc(sizeof(float)*(k*n)); 
    float* V_h = (float*)malloc(sizeof(float)*(k*n)); 
    float* C_h = (float*)malloc(sizeof(float)*(m*n));

    for(int i =0; i< m*k;i++){
        A_h[i] = rand()%100/100.0;
    }
    
    for(int i =0; i< k*n;i++){
        B_h[i] = rand()%100/100.0;
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    for(int p = 0; p <m; p++){
        for(int i =0; i< m;i++){
            for(int j = 0; j< n; j++){
                V_h [i+(p*n)]+= A_h[j + (p*k)] * B_h[i + (j*k)];
            }
        }
    }
    
    for(int p = 0; p <m; p++){
        for(int i =0; i< m;i++){
            for(int j = 0; j< n; j++){
                C_h [i+(p*n)]+= A_h[j + (p*k)] * B_h[i + (j*k)];
            }
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    float time_to_add_seconds = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1e6f;

    printf("VeccMult on CPU: %25s%9.6fs \n", "", time_to_add_seconds);

    start_time = std::chrono::high_resolution_clock::now();
    basicSgemm_d_1thread1element(m,k,n,A_h,B_h,C_h);
    end_time = std::chrono::high_resolution_clock::now();
    time_to_add_seconds = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1e6f;
    printf("matrixMulKernel_1thread1row on GPU: %25s%9.6fs \n", "", time_to_add_seconds);


    start_time = std::chrono::high_resolution_clock::now();
    basicSgemm_tiled(m,k,n,A_h,B_h,C_h,V_h);
    end_time = std::chrono::high_resolution_clock::now();
    time_to_add_seconds = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1e6f;
    printf("matrixMulKernel_tiled on GPU: %25s%9.6fs \n", "", time_to_add_seconds);

    if(verify_result(A_h,B_h,C_h,V_h,n,m) == true){
        printf("Verifying result...TEST PASSED! \n");
    }

    free(A_h);
    free(B_h);
    free(C_h);
  return 0;
}