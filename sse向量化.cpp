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
        __m128 vt = _mm_set1_ps(A[k][k]);
        for (int i = k + 1; i < n; i += 4) {
            int endi = min(i + 4, n);
            __m128 va = _mm_loadu_ps(A[k] + i);
            __m128 vfactor = _mm_div_ps(va, vt);

            for (int j = k + 1; j < n; j += 4) {
                int endj = j + 4 > n ? n : j + 4; // 计算当前循环迭代的结束点  
                __m128 Aj = _mm_loadu_ps(&A[i][j]); // 加载A[i][j]到j+3的4个浮点数到向量寄存器  
                __m128 Ak = _mm_loadu_ps(&A[k][j]); // 加载A[k][j]到j+3的4个浮点数到向量寄存器    

                // 计算 A[i][j] -= factor * A[k][j] 的SSE版本  
                __m128 result = _mm_sub_ps(Aj, _mm_mul_ps(vfactor, Ak));

                // 将结果存回A[i][j]到j+3  
                _mm_storeu_ps(&A[i][j], result);
            }

            // 更新b[i]，由于b是一维数组，我们不需要考虑多个元素的情况  
            // 直接进行单个元素的计算即可  

            __m128 bk = _mm_set1_ps(b[k]);
            __m128 factor = _mm_mul_ps(vfactor, bk);
            __m128 bi = _mm_loadu_ps(b + i);
            bi = _mm_sub_ps(bi, factor);
            _mm_storeu_ps(b + i, bi);
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

    for (int i = 0; i < n; i++)
    {
        cout << x[i];
    }

    // 计算时间  
    gettimeofday(&end, NULL);
    double timecount = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0;
    cout << "Time taken: " << timecount << " ms" << endl;
    return 0;
}

