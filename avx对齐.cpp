#include <sys/time.h>  
#include <iostream>  
#include <immintrin.h> // AVX指令集  
#include <cstdlib> // for rand() and srand()  

using namespace std;

int main() {
    const int n = 1000;
    const int alignment = 32; // AVX通常要求16或32字节对齐  

    // 使用_mm_malloc分配对齐的内存  
    float* A_aligned = (float*)_mm_malloc(n * n * sizeof(float), alignment);
    float* b_aligned = (float*)_mm_malloc(n * sizeof(float), alignment);
    float* x_aligned = (float*)_mm_malloc(n * sizeof(float), alignment);

    float(*A)[n] = reinterpret_cast<float(*)[n]>(A_aligned);
    float* b = b_aligned;
    float* x = x_aligned;

    srand(time(NULL)); // 初始化随机数生成器  

    // 填充A和b数组...  
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i][j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 10.0 - 5.0;
        }
        b[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 10.0 - 5.0;
    }

    struct timeval start, end;
    gettimeofday(&start, NULL);

    // 使用AVX指令集进行高斯消元法的部分计算  
    for (int k = 0; k < n - 1; ++k) {
        __m256 vt = _mm256_set1_ps(A[k][k]);

        for (int i = k + 1; i < n; i += 8) { // AVX每次处理8个float  
            int endi = min(i + 8, n);

            __m256 va = _mm256_loadu_ps(A[k] + i);

            __m256 vfactor = _mm256_div_ps(va, vt);

            for (int j = k + 1; j < n; j += 8) {
                int endj = min(j + 8, n);

                __m256 Aj = _mm256_loadu_ps(A[i] + j);
                __m256 Ak = _mm256_loadu_ps(A[k] + j);

                __m256 result = _mm256_sub_ps(Aj, _mm256_mul_ps(vfactor, Ak));

                _mm256_storeu_ps(A[i] + j, result);
            }

            __m256 bk = _mm256_set1_ps(b[k]);
            __m256 factor = _mm256_mul_ps(vfactor, bk);

            __m256 bi = _mm256_loadu_ps(b + i);
            bi = _mm256_sub_ps(bi, factor);
            _mm256_storeu_ps(b + i, bi);
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


    // 使用_mm_free释放对齐的内存  
    _mm_free(A_aligned);
    _mm_free(b_aligned);
    _mm_free(x_aligned);

    // 计算时间  
    gettimeofday(&end, NULL);
    double timecount = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0;
    cout << "Time taken: " << timecount << " ms" << endl;


    return 0;
}