#include <cuda.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

__global__ void blelloch_scan(float *g_data, int n) {
    extern __shared__ float temp[];

    int thid = threadIdx.x;
    int offset = 1;

    // Carrega dados na memória compartilhada
    int ai = thid * 2;
    if (ai < n) {
        temp[ai] = g_data[ai];
        if (ai + 1 < n)
            temp[ai + 1] = g_data[ai + 1];
    }
    __syncthreads();

    // Fase de upsweep (redução)
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    // Limpa o último elemento para a fase de downsweep
    if (thid == 0) {
        temp[n - 1] = 0;
    }

    // Fase de downsweep
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    // Escreve os resultados de volta na memória global
    if (ai < n) {
        g_data[ai] = temp[ai];
        if (ai + 1 < n)
            g_data[ai + 1] = temp[ai + 1];
    }
}

void run_blelloch_scan(std::vector<float>& data) {
    float *d_data;
    int n = data.size();
    size_t size = n * sizeof(float);

    // Aloca memória na GPU
    cudaMalloc((void**)&d_data, size);
    cudaMemcpy(d_data, data.data(), size, cudaMemcpyHostToDevice);

    // Definindo block e grid sizes
    int blockSize = 512;
    int gridSize = (n + blockSize - 1) / blockSize;
    int sharedMemSize = blockSize * 2 * sizeof(float);

    // Executa o kernel CUDA
    auto start = std::chrono::high_resolution_clock::now();
    blelloch_scan<<<gridSize, blockSize, sharedMemSize>>>(d_data, n);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    // Copia o resultado de volta para a CPU
    cudaMemcpy(data.data(), d_data, size, cudaMemcpyDeviceToHost);

    // Limpa a memória da GPU
    cudaFree(d_data);

    // Imprime o tempo de execução com 6 casas decimais de precisão
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Array Size: " << std::setw(10) << n
              << ", Time Taken: " << std::fixed << std::setprecision(6)
              << elapsed.count() << " seconds" << std::endl;
}

int main() {
    std::vector<int> sizes = {100, 1000, 10000, 100000, 1000000, 10000000};
    for (int size : sizes) {
        std::vector<float> data(size);
        for (int i = 0; i < size; ++i) {
            data[i] = static_cast<float>(rand()) / RAND_MAX;
        }
        run_blelloch_scan(data);
    }
    return 0;
}
