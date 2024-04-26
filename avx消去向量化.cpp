#include<sys/time.h>
#include<iostream>  
#include<stdio.h>
#include<stdlib.h>
#include <emmintrin.h>
#include<immintrin.h>
using namespace std;
int main() {
    const int n = 200;
    float A[n][n];
    float b[n], x[n];
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            A[i][j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 10.0 - 5.0;
        }
    }
    for (int i = 0; i < n; i++)
    {
        b[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 10.0 - 5.0;
    }

    struct timeval start;
    struct timeval end;
    gettimeofday(&start, NULL);
    float timecount = 0;

    for (int k = 0; k < n; k++) {
        for (int i = k + 1; i < n; i++) {
            double factor = A[i][k] / A[k][k];
            for (int j = k + 1; j < n; j += 8) {
                int endj = j + 8 > n ? n : j + 8; // 计算当前循环迭代的结束点  
                __m256 Aj = _mm256_loadu_ps(&A[i][j]); // 加载A[i][j]到j+3的4个浮点数到向量寄存器  
                __m256 Ak = _mm256_loadu_ps(&A[k][j]); // 加载A[k][j]到j+3的4个浮点数到向量寄存器  
                __m256 factorVec = _mm256_set1_ps(factor); // 创建一个包含factor的向量  

                // 计算 A[i][j] -= factor * A[k][j] 的SSE版本  
                __m256 result = _mm256_sub_ps(Aj, _mm256_mul_ps(factorVec, Ak));

                // 将结果存回A[i][j]到j+3  
                _mm256_storeu_ps(&A[i][j], result);
            }

            // 更新b[i]，由于b是一维数组，我们不需要考虑多个元素的情况  
            // 直接进行单个元素的计算即可  
            b[i] -= factor * b[k];
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

    gettimeofday(&end, NULL);
    timecount += (end.tv_sec - start.tv_sec) * 1000000 + end.tv_usec - start.tv_usec;
    cout << timecount << endl;
    return 0;
}