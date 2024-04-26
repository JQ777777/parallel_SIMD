#include <sys/time.h>
#include <iostream>
#include <immintrin.h>
using namespace std;

int main() {
    const int n = 200;
    float A[n][n];
    float b[n], x[n];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 10.0 - 5.0;
        }
    }
    for (int i = 0; i < n; i++) {
        b[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 10.0 - 5.0;
    }

    struct timeval start;
    struct timeval end;
    gettimeofday(&start, NULL);


    for (int k = 0; k < n - 1; k++) {
        __m256 vt = _mm256_set1_ps(A[k][k]);
        for (int i = k + 1; i < n; i += 8) {
            int endi = min(i + 8, n);
            __m256 va = _mm256_loadu_ps(A[k] + i);
            __m256 vfactor = _mm256_div_ps(va, vt);

            // 更新b向量  
            float vb[8];
            for (int p = 0; p < 4; ++p) {
                vb[p] = b[i + p];
            }

            float vbk = b[k];
            float factor; // 假设vfactor是之前计算好的一个值  
            float vtemp;

            for (int p = 0; p < 8; ++p) {
                vtemp = factor * vbk;
                vb[p] -= vtemp;
            }

            // 将修改后的值写回b向量  
            for (int p = 0; p < 8; ++p) {
                b[i + p] = vb[p];
            }

            // 更新矩阵A  
            for (int j = k + 1; j < n; ++j) {
                float Akj = A[k][j];
                float Aji;
                for (int p = 0; p < 8 && i + p < n; ++p) {
                    Aji = A[j][i + p];
                    Aji -= factor * Akj;
                    A[j][i + p] = Aji;
                }
            }
        }
    }

    // 回代过程  
    x[n - 1] = b[n - 1] / A[n - 1][n - 1];
    for (int i = n - 2; i >= 0; i--) {
        double sum = b[i];
        for (int j = i + 1; j < n; j++) {
            sum -= A[i][j] * x[j];
        }
        x[i] = sum / A[i][i];
    }

    // 计算时间  
    gettimeofday(&end, NULL);
    double timecount = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0;
    cout << "Time taken: " << timecount << " ms" << endl;
    return 0;
}
