#include <opencv2/opencv.hpp>
#include <string>

__constant__ float Kernel [5*5];

cv::Mat CPU_conv(cv::Mat loadedImg){
    cv::Mat kernelCPU = cv::Mat::ones(5,5,CV_32F) / 25.0;
    cv::Mat blurredImage;
    cv::filter2D(loadedImg, blurredImage, -1, kernelCPU);
    return blurredImage;
}

__global__ void ImageConvKernel(float *ImageIn, float *ImageOut,int width, int hight){
    unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;

    int r = 5/2;
    float sum = 0.0f;

    for(int i = 0; i < 5;i++){
        for(int j = 0; j < 5;j++){
            int t = (((row-r)+i)*(width));
            int p = (col - r) + j;
            sum += ImageIn[t + p] * Kernel[(i*5)+j];
        }
    }

    ImageOut[row* width + col] = sum;
}

__global__ void ImageConvTiling(float *ImageIn, float *ImageOut,int width, int height){
        extern __shared__ float sharedMem[]; // Dynamic shared memory

    const int TILE_WIDTH = 16; // Tile size
    const int KERNEL_RADIUS = 5 / 2;
    int sharedWidth = TILE_WIDTH + 2 * KERNEL_RADIUS;

    // Calculate thread and block indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_o = blockIdx.y * TILE_WIDTH + ty;
    int col_o = blockIdx.x * TILE_WIDTH + tx;
    int row_i = row_o - KERNEL_RADIUS;
    int col_i = col_o - KERNEL_RADIUS;

    if (row_i >= 0 && row_i < height && col_i >= 0 && col_i < width) {
        sharedMem[ty * sharedWidth + tx] = ImageIn[row_i * width + col_i];
    } else {
        sharedMem[ty * sharedWidth + tx] = 0.0f; // Zero-padding
    }
    __syncthreads();

    // Perform convolution for valid threads
    float value = 0.0f;
    if (ty < TILE_WIDTH && tx < TILE_WIDTH) {
        for (int ky = 0; ky < 5; ky++) {
            for (int kx = 0; kx < 5; kx++) {
                int smRow = ty + ky;
                int smCol = tx + kx;
                value += sharedMem[smRow * sharedWidth + smCol] * Kernel[ky * 5 + kx];
            }
        }
        // Write output if within bounds
        if (row_o < height && col_o < width) {
            ImageOut[row_o * width + col_o] = value;
        }
    }
    __syncthreads();
}

cv::Mat GPU_conv(cv::Mat loadedImg){
    int nRows = loadedImg.rows, nCols = loadedImg.cols;

    float localKernel[5*5];
    for(int i = 0; i <25; i++){
        localKernel[i] = 1.0f/25.0f;
    }
    auto start_time = std::chrono::high_resolution_clock::now();
    cudaMemcpyToSymbol(Kernel, localKernel, (5*5)*sizeof(float));

    cv::Mat floatImg;
    loadedImg.convertTo(floatImg, CV_32F);
    float *ImageIn = nullptr;
    float *ImageOut = nullptr;
    cudaMalloc(&ImageIn,sizeof(float)*(nRows*nCols));
    cudaMalloc(&ImageOut,sizeof(float)*(nRows*nCols));
    auto end_time = std::chrono::high_resolution_clock::now();
    float time_to_add_seconds = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1e6f;

    printf("    cudaMalloc: %24s%9.6fs \n", "",time_to_add_seconds);

    start_time = std::chrono::high_resolution_clock::now();
    cudaMemcpy(ImageIn, floatImg.data, sizeof(float)*(nRows*nCols), cudaMemcpyKind::cudaMemcpyHostToDevice);
    end_time = std::chrono::high_resolution_clock::now();
    time_to_add_seconds = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1e6f;
    
    printf("    cudaMemcpy: %24s%9.6fs \n", "", time_to_add_seconds);

    start_time = std::chrono::high_resolution_clock::now();
    dim3 blockDim(32,32,1);
    dim3 gridDim(32,32,1);
    ImageConvKernel<<<gridDim,blockDim>>>(ImageIn,ImageOut,nRows,nCols);
    cudaDeviceSynchronize();
    end_time = std::chrono::high_resolution_clock::now();
    time_to_add_seconds = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1e6f;
    printf("    ImageConvKernel<<<(%d,%d,%d),(%d,%d,%d)>>>: %9.6f \n",32,32,1,32,32,1,  time_to_add_seconds);
    cv::Mat fImageOut = cv::Mat::zeros(nRows, nCols, CV_32F);
    start_time = std::chrono::high_resolution_clock::now();
    cudaMemcpy(fImageOut.data,ImageOut,sizeof(float)*(nRows*nCols),cudaMemcpyKind::cudaMemcpyDeviceToHost);
    end_time = std::chrono::high_resolution_clock::now();
    time_to_add_seconds = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1e6f;
    printf("    CudaMemcpy: %24s%9.6fs \n", "", time_to_add_seconds);
    cv::Mat outputImage;
    fImageOut.convertTo(outputImage, CV_8U);
    
    cudaFree(ImageIn);
    cudaFree(ImageOut);
    return outputImage;

}

cv::Mat GPU_conv_tiling(cv::Mat loadedImg){
    int nRows = loadedImg.rows, nCols = loadedImg.cols;

    float localKernel[5*5];
    for(int i = 0; i <25; i++){
        localKernel[i] = 1.0f/25.0f;
    }
    auto start_time = std::chrono::high_resolution_clock::now();
    cudaMemcpyToSymbol(Kernel, localKernel, (5*5)*sizeof(float));

    cv::Mat floatImg;
    loadedImg.convertTo(floatImg, CV_32F);

    float *ImageIn = nullptr;
    float *ImageOut = nullptr;
    cudaMalloc(&ImageIn,sizeof(float)*(nRows*nCols));
    cudaMalloc(&ImageOut,sizeof(float)*(nRows*nCols));
    auto end_time = std::chrono::high_resolution_clock::now();
    float time_to_add_seconds = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1e6f;

    printf("    cudaMalloc: %24s%9.6fs \n", "",time_to_add_seconds);

    start_time = std::chrono::high_resolution_clock::now();
    cudaMemcpy(ImageIn, floatImg.data, sizeof(float)*(nRows*nCols), cudaMemcpyKind::cudaMemcpyHostToDevice);
    end_time = std::chrono::high_resolution_clock::now();
    time_to_add_seconds = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1e6f;
    
    printf("    cudaMemcpy: %24s%9.6fs \n", "", time_to_add_seconds);

    start_time = std::chrono::high_resolution_clock::now();
    dim3 blockDim((16 + (2 * (5 / 2))),(16 + 2 * (5 / 2)),1);
    dim3 gridDim((nRows + 16- 1) / 16,
                  (nCols + 16 - 1) / 16);
    size_t sharedDataSize = (16 + 2 * (5 / 2)) * (16 + 2 * (5 / 2)) * sizeof(float);
    ImageConvTiling<<<gridDim,blockDim, sharedDataSize>>>(ImageIn,ImageOut,nRows,nCols);
    cudaDeviceSynchronize();
    end_time = std::chrono::high_resolution_clock::now();
    time_to_add_seconds = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1e6f;
    printf("    ImageConvTiling<<<(%d,%d,%d),(%d,%d,%d),(%d)>>>: %9.6f \n",(16 + 2 * (5 / 2)),(16 + 2 * (5 / 2)),1,(nRows + 16- 1) / 16,(nCols + 16 - 1) / 16,
    1, (16 + 2 * (5 / 2)) * (16 + 2 * (5 / 2)) * sizeof(float), time_to_add_seconds);
    cv::Mat fImageOut = cv::Mat::zeros(nRows, nCols, CV_32F);
    start_time = std::chrono::high_resolution_clock::now();
    cudaMemcpy(fImageOut.data,ImageOut,sizeof(float)*(nRows*nCols),cudaMemcpyKind::cudaMemcpyDeviceToHost);
    end_time = std::chrono::high_resolution_clock::now();
    time_to_add_seconds = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1e6f;
    printf("    CudaMemcpy: %24s%9.6fs \n", "", time_to_add_seconds);
    cv::Mat outputImage;
    fImageOut.convertTo(outputImage, CV_8U);
    
    cudaFree(ImageIn);
    cudaFree(ImageOut);
    return outputImage;

}

bool verify(cv::Mat answer1, cv::Mat answer2, unsigned int nRows, unsigned int nCols){
        for (int i = 0; i < answer1.rows; ++i) {
        for (int j = 0; j < answer1.cols; ++j) {
            // Compare the pixel values in the single channel (grayscale)
            uchar pixel1 = answer1.at<uchar>(i, j);
            uchar pixel2 = answer2.at<uchar>(i, j);

            // Check if the difference between the pixel values is within the tolerance
            if (std::abs(pixel1 - pixel2) > 10) {
                return false;  // If the difference exceeds the tolerance, return false
            }
        }
    }

    return true;  // If all pixels are within the tolerance, return true
}

int main(int argc, char** argv){
    cudaDeviceSynchronize();
    char* path = argv[1];
    cv::Mat loadedImg = cv::imread(path);
    if(loadedImg.empty()) return -1;

    cv::Size kernelSize(5, 5);

    cv::Mat blurredImg_opencv;
    cv::blur(loadedImg, blurredImg_opencv, kernelSize);

    auto start_time = std::chrono::high_resolution_clock::now();
    cv::Mat blurredImg_cpu = CPU_conv(loadedImg);
    auto end_time = std::chrono::high_resolution_clock::now();
    float time_to_add_seconds = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1e6f;

    printf("blurredImg_cpu on CPU: %25s%9.6fs \n", "", time_to_add_seconds);
    verify(blurredImg_cpu, blurredImg_opencv, loadedImg.rows, loadedImg.cols);

    cv::Mat loadedImgGray;
    cv::cvtColor(loadedImg, loadedImgGray, cv::COLOR_BGR2GRAY);
    start_time = std::chrono::high_resolution_clock::now();
    cv::Mat blurredImg_gpu = GPU_conv(loadedImgGray);
    end_time = std::chrono::high_resolution_clock::now();
    time_to_add_seconds = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1e6f;
    printf("blurredImg_gpu on GPU: %25s%9.6fs \n", "", time_to_add_seconds);
    verify(blurredImg_gpu, blurredImg_opencv, loadedImg.rows, loadedImg.cols);

    start_time = std::chrono::high_resolution_clock::now();
    cv::Mat blurredImg_tiled_gpu = GPU_conv_tiling(loadedImgGray);
    end_time = std::chrono::high_resolution_clock::now();
    time_to_add_seconds = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1e6f;
    printf("blurredImg_tiled_gpu on GPU: %25s%9.6fs \n", "", time_to_add_seconds);
    verify(blurredImg_tiled_gpu, blurredImg_opencv, loadedImg.rows, loadedImg.cols);

    cv::imwrite("blurredImg_opencv.jpg",blurredImg_opencv);
    cv::imwrite("blurredImg_cpu.jpg",blurredImg_cpu);
    cv::imwrite("blurredImg_gpu.jpg",blurredImg_gpu);
    cv::imwrite("blurredImg_tiled_gpu.jpg",blurredImg_tiled_gpu);
}