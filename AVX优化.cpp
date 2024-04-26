#include <sys/time.h>  
#include <iostream>  
#include <cstdlib>  
#include <immintrin.h>  
#include <cmath>  

using namespace std;

int main()
{
    const int n = 200;
    float A[n][n] __attribute__((aligned(32))); // 确保矩阵A按32字节对齐  
    float b[n] __attribute__((aligned(32)));     // 确保向量b按32字节对齐  
    float x[n];

    srand(time(NULL));

    // 初始化矩阵A和向量b  
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i][j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 10.0f - 5.0f;
        }
        b[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 10.0f - 5.0f;
    }

    struct timeval start, end;
    gettimeofday(&start, nullptr);

    for (int k = 0; k < n - 1; ++k) {
        // Partial pivoting is omitted for simplicity  
        float pivot = A[k][k];
        if (fabs(pivot) < 1e-10f) {
            cerr << "Error: zero pivot found at position (" << k << ", " << k << ")" << endl;
            return -1;
        }

        // Scale row k by 1/pivot  
        for (int i = k; i < n; i += 8) {
            __m256 vt = _mm256_set1_ps(1.0f / pivot);
            __m256 va = _mm256_loadu_ps(A[k] + i);
            __m256 vi = _mm256_mul_ps(va, vt);
            _mm256_storeu_ps(A[k] + i, vi);
        }

        // Update vector b  
        for (int i = k + 1; i < n; ++i) {
            b[i] -= A[i][k] * b[k];
        }

        // Update the rest of the matrix  
        for (int i = k + 1; i < n; ++i) {
            for (int j = k + 1; j < n; j += 8) {
                __m256 vfactor = _mm256_loadu_ps(A[i] + k);
                __m256 vAj = _mm256_loadu_ps(A[j] + i);
                __m256 vtemp = _mm256_mul_ps(vfactor, vAj);
                vAj = _mm256_sub_ps(vAj, vtemp);
                _mm256_storeu_ps(A[j] + i, vAj);
            }
        }
    }

    // Back substitution to solve for x  
    for (int i = n - 1; i >= 0; --i) {
        float sum = b[i];
        for (int j = i + 1; j < n; ++j) {
            sum -= A[i][j] * x[j];
        }
        x[i] = sum / A[i][i];
    }

    gettimeofday(&end, nullptr);
    long long timecount = (end.tv_sec - start.tv_sec) * 1000000LL + end.tv_usec - start.tv_usec;
    cout << "Time: " << timecount << " microseconds" << endl;

    return 0;
}