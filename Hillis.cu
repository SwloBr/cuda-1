#include <cuda.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

__global__ void hillis_steele_scan(float *g_data, int n, int *step_count) {
    int thid = threadIdx.x + blockIdx.x * blockDim.x;

    // Hillis-Steele Prefix Sum
    int steps = 0; // Contador de passos
    for (int offset = 1; offset < n; offset *= 2) {
        if (thid >= offset && thid < n) {
            g_data[thid] += g_data[thid - offset];
        }
        __syncthreads(); // Synchronize all threads before proceeding to next offset
        steps++;
    }

    // Atualiza a contagem de passos (apenas uma thread escreve no final para evitar corrida de dados)
    if (thid == 0) {
        atomicAdd(step_count, steps);
    }
}

void run_hillis_steele_scan(std::vector<float>& data) {
    float *d_data;
    int *d_step_count;
    int n = data.size();
    size_t size = n * sizeof(float);

    // Aloca memória na GPU
    cudaMalloc((void**)&d_data, size);
    cudaMemcpy(d_data, data.data(), size, cudaMemcpyHostToDevice);

    // Aloca memória para contagem dos passos
    cudaMalloc((void**)&d_step_count, sizeof(int));
    cudaMemset(d_step_count, 0, sizeof(int)); // Inicializa com 0

    // Definindo block e grid sizes
    int blockSize = 512;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Repita a execução do kernel várias vezes
    int repeat = 100;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repeat; ++i) {
        hillis_steele_scan<<<gridSize, blockSize>>>(d_data, n, d_step_count);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    // Copia o resultado de volta para a CPU
    cudaMemcpy(data.data(), d_data, size, cudaMemcpyDeviceToHost);

    // Copia a contagem dos passos para a CPU
    int step_count;
    cudaMemcpy(&step_count, d_step_count, sizeof(int), cudaMemcpyDeviceToHost);

    // Limpa a memória da GPU
    cudaFree(d_data);
    cudaFree(d_step_count);

    // Imprime o tempo de execução médio e o número de passos
    std::chrono::duration<double> elapsed = end - start;
    double average_time = elapsed.count() / repeat;
    std::cout << "Array Size: " << std::setw(10) << n
              << ", Average Time Taken: " << std::fixed << std::setprecision(6)
              << average_time << " seconds"
              << ", Steps: " << step_count << std::endl;
}

int main() {
    std::vector<int> sizes = {100, 1000, 10000, 100000, 1000000, 10000000};
    for (int size : sizes) {
        std::vector<float> data(size);
        for (int i = 0; i < size; ++i) {
            data[i] = static_cast<float>(rand()) / RAND_MAX;
        }
        run_hillis_steele_scan(data);
    }
    return 0;
}
