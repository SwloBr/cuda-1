#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

void run_reduce_serial(std::vector<float>& data) {
    int n = data.size();
    int steps = 0;
    float total = 0.0f;

    // Medindo o tempo de execução
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n; ++i) {
        total += data[i];
        steps++; // Contando o número de passos (operações)
    }

    auto end = std::chrono::high_resolution_clock::now();

    // Imprime o resultado
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Array Size: " << std::setw(10) << n
              << ", Total Sum: " << std::fixed << std::setprecision(6) << total
              << ", Time Taken: " << std::fixed << std::setprecision(6)
              << elapsed.count() << " seconds"
              << ", Steps: " << steps << std::endl;
}

int main() {
    std::vector<int> sizes = {100, 1000, 10000, 100000, 1000000, 10000000};
    for (int size : sizes) {
        std::vector<float> data(size);
        for (int i = 0; i < size; ++i) {
            data[i] = static_cast<float>(rand()) / RAND_MAX;
        }
        run_reduce_serial(data);
    }
    return 0;
}
