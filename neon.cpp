#include<sys/time.h>
#include<iostream>  
#include<stdio.h>
#include<stdlib.h>
#include <arm_neon.h>
using namespace std;

void matrix_operation(float* A, int n) {
    for (int k = 0; k < n - 1; k++) {   
        float32x4_t vt = vld1q_f32(&A[k * n + k]); 
        for (int j = k + 1; j <= n - 4; j += 4) {   
            float32x4_t va = vld1q_f32(&A[k * n + j]);  
            va = vmulq_f32(va, vrecpeq_f32(vt));    
            vst1q_f32(&A[k * n + j], va);
        }
        for (int j = (k + 1 + (n - 1) / 4) * 4; j < n; j++) {
            A[k * n + j] /= A[k * n + k];
        }
        A[k * n + k] = 1.0f;
        for (int i = k + 1; i < n; i++) {    
            float32x4_t vaik = vdupq_n_f32(A[i * n + k]);
            for (int j = k + 1; j <= n - 4; j += 4) {  
                float32x4_t vakj = vld1q_f32(&A[k * n + j]); 
                float32x4_t vaij = vld1q_f32(&A[i * n + j]);
                float32x4_t vx = vmulq_f32(vakj, vaik);
                vaij = vsubq_f32(vaij, vx); 
                vst1q_f32(&A[i * n + j], vaij);
            }
            for (int j = (k + 1 + (n - 1) / 4) * 4; j < n; j++) {
                A[i * n + j] -= A[k * n + j] * A[i * n + k];
            }
            A[i * n + k] = 0.0f;
        }
    }
}

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

    matrix_operation(*A, n);

    gettimeofday(&end, NULL);
    timecount += (end.tv_sec - start.tv_sec) * 1000000 + end.tv_usec - start.tv_usec;
    cout << timecount << endl;
    return 0;
}