#include<sys/time.h>
#include<iostream>  
#include<stdio.h>
#include<stdlib.h>
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
            for (int j = k + 1; j < n; j++) {
                A[i][j] -= factor * A[k][j];
            }
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