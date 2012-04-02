#include <emmintrin.h>
#include <stdio.h>

void pad ( int n, int nt, float *dst, float *src );
void unpad ( int n, int nt, float *dst, float *src );
void square_sgemm_norm (int n, float* A, float* B, float* C);
void square_sgemm_less (int n, float* A, float* B, float* C);
void square_sgemm_plus (int n, float* A, float* B, float* C);

/* Break into three cases. Where 
   n is divisible by 32, where n
   is one more than 32, and where
   n is one less than 32          */
void square_sgemm (int n, float* A, float* B, float* C) {
  int remain = n % 32;
  if (remain == 0) {
    square_sgemm_norm(n, A, B, C);
  } else if (remain == 1) {
    square_sgemm_plus(n, A, B, C);
  } else {
    square_sgemm_less(n, A, B, C);
  }
}

/* Used to pad matrices moving them from 
   src to dst where n is the size of the
   src and nt is the size of the dst (nt>n) */
void pad ( int n, int nt, float *dst, float *src ) {
  int blocksize = 384;
  if (n < blocksize) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        dst[j+i*nt] = src[j+i*n];
      }
    }
  } else {
    for (int x = 0; x < n; x += blocksize) {
      for (int y = 0; y < n; y += blocksize) {
        int xMax = x + blocksize;
        if (x > n - blocksize) {
          xMax = n;
        }
        for (int i = x; i < xMax; i++) {
          int yMax = y + blocksize;
          if (y > n - blocksize) {
            yMax = n;
          }
          for (int j = y; j < yMax; j++) {
            dst[j+i*nt] = src[j+i*n];
          }
        }
      }
    }
  }
}

/* Used to unpad matrices moving them from
   src to dst where nt is the size of the
   src and nt is the size of the dst (nt>n) */
void unpad ( int n, int nt, float *dst, float *src ) {
  int blocksize = 384;
  if (n < blocksize) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        dst[j+i*n] = src[j+i*nt];
      }
    }
  } else {
    for (int x = 0; x < n; x += blocksize) {
      for (int y = 0; y < n; y += blocksize) {
        int xMax = x + blocksize;
        if (x > n - blocksize) {
          xMax = n;
        }
        for (int i = x; i < xMax; i++) {
          int yMax = y + blocksize;
          if (y > n - blocksize) {
            yMax = n;
          }
          for (int j = y; j < yMax; j++) {
            dst[j+i*n] = src[j+i*nt];
          }
        }
      }
    }
  }
}

/* square_sgemm_norm accurately handles
   matrices with n divisible by 32.    */
void square_sgemm_norm (int n, float* A, float* B, float* C) {
  int padsize;
  __m128 a1;
  __m128 b1;
  __m128 c1,c2,c3,c4,c5,c6,c7,c8;
  // ensures optimal blocksize based on n
  if (n < 129) {
    padsize = 32;
  } else {
    padsize = 5120/n;
  }
  for (int w = 0; w < n; w += padsize) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < (n/32)*32; j += 32) {
        int xMax = w + padsize;
        if (w > n - padsize) {
          xMax = n;
        }
        for (int k = w; k < xMax; k += 1) {
          b1 = _mm_load1_ps(&B[k + i*n]);

          a1 = _mm_loadu_ps(&A[j + k*n]);
          c1 = _mm_add_ps(c1, _mm_mul_ps(a1, b1));
          a1 = _mm_loadu_ps(&A[j + k*n + 4]);
          c2 = _mm_add_ps(c2, _mm_mul_ps(a1, b1));
          a1 = _mm_loadu_ps(&A[j + k*n + 8]);
          c3 = _mm_add_ps(c3, _mm_mul_ps(a1, b1));
          a1 = _mm_loadu_ps(&A[j + k*n + 12]);
          c4 = _mm_add_ps(c4, _mm_mul_ps(a1, b1));
          a1 = _mm_loadu_ps(&A[j + k*n + 16]);
          c5 = _mm_add_ps(c5, _mm_mul_ps(a1, b1));
          a1 = _mm_loadu_ps(&A[j + k*n + 20]);
          c6 = _mm_add_ps(c6, _mm_mul_ps(a1, b1));
          a1 = _mm_loadu_ps(&A[j + k*n + 24]);
          c7 = _mm_add_ps(c7, _mm_mul_ps(a1, b1));
          a1 = _mm_loadu_ps(&A[j + k*n + 28]);
          c8 = _mm_add_ps(c8, _mm_mul_ps(a1, b1));
        }

        _mm_storeu_ps(&C[j + i*n], _mm_add_ps(c1, _mm_loadu_ps(&C[j + i*n])));
        _mm_storeu_ps(&C[j + i*n + 4], _mm_add_ps(c2, _mm_loadu_ps(&C[j + i*n + 4])));
        _mm_storeu_ps(&C[j + i*n + 8], _mm_add_ps(c3, _mm_loadu_ps(&C[j + i*n + 8])));
        _mm_storeu_ps(&C[j + i*n + 12], _mm_add_ps(c4, _mm_loadu_ps(&C[j + i*n + 12])));
        _mm_storeu_ps(&C[j + i*n + 16], _mm_add_ps(c5, _mm_loadu_ps(&C[j + i*n + 16])));
        _mm_storeu_ps(&C[j + i*n + 20], _mm_add_ps(c6, _mm_loadu_ps(&C[j + i*n + 20])));
        _mm_storeu_ps(&C[j + i*n + 24], _mm_add_ps(c7, _mm_loadu_ps(&C[j + i*n + 24])));
        _mm_storeu_ps(&C[j + i*n + 28], _mm_add_ps(c8, _mm_loadu_ps(&C[j + i*n + 28])));

        c1 = _mm_setzero_ps();
        c2 = _mm_setzero_ps();
        c3 = _mm_setzero_ps();
        c4 = _mm_setzero_ps();
        c5 = _mm_setzero_ps();
        c6 = _mm_setzero_ps();
        c7 = _mm_setzero_ps();
        c8 = _mm_setzero_ps();
      }
    }
  }
}

/* handles the case where n is one less 
   than a multiple of 32 */
void square_sgemm_less (int n, float* A, float* B, float* C) {
  float *At, *Bt, *Ct;
  int nt = n+1;

  //load into buffers for optimization
  float *bufferA = (float*) calloc((nt)*(nt), sizeof(float));
  At = bufferA;
  float *bufferB = (float*) calloc((nt)*(nt), sizeof(float));
  Bt = bufferB;
  float *bufferC = (float*) calloc((nt)*(nt), sizeof(float));
  Ct = bufferC;

  //pad each matrix
  pad(n,nt,At,A);
  pad(n,nt,Bt,B);
  pad(n,nt,Ct,C);

  //run as if divisible by 32
  square_sgemm_norm (nt, At, Bt, Ct);

  //unpad Ct for correct results in C matrix
  unpad(n,nt,C,Ct);
}

void square_sgemm_plus (int n, float* A, float* B, float* C) {
  int blocksize, xMax;
  __m128 a1;
  __m128 b1;
  __m128 c1,c2,c3,c4,c5,c6,c7,c8;
  if(n < 129) {
    blocksize = 32;
  } else {
    blocksize = 5120/n;
  }
  for (int kk = 0; kk < n; kk += blocksize) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < (n/32)*32; j += 32) {
        xMax = kk + blocksize;
        if (kk > n - blocksize) {
          xMax = n;
        }
        for (int k = kk; k < xMax; k += 1) {

          b1 = _mm_load1_ps(&B[k + i*n]);
          a1 = _mm_loadu_ps(&A[j + k*n]);
          c1 = _mm_add_ps(c1, _mm_mul_ps(a1, b1));
          a1 = _mm_loadu_ps(&A[j + k*n + 4]);
          c2 = _mm_add_ps(c2, _mm_mul_ps(a1, b1));
          a1 = _mm_loadu_ps(&A[j + k*n + 8]);
          c3 = _mm_add_ps(c3, _mm_mul_ps(a1, b1));
          a1 = _mm_loadu_ps(&A[j + k*n + 12]);
          c4 = _mm_add_ps(c4, _mm_mul_ps(a1, b1));
          a1 = _mm_loadu_ps(&A[j + k*n + 16]);
          c5 = _mm_add_ps(c5, _mm_mul_ps(a1, b1));
          a1 = _mm_loadu_ps(&A[j + k*n + 20]);
          c6 = _mm_add_ps(c6, _mm_mul_ps(a1, b1));
          a1 = _mm_loadu_ps(&A[j + k*n + 24]);
          c7 = _mm_add_ps(c7, _mm_mul_ps(a1, b1));
          a1 = _mm_loadu_ps(&A[j + k*n + 28]);
          c8 = _mm_add_ps(c8, _mm_mul_ps(a1, b1));
        }

        _mm_storeu_ps(&C[j + i*n], _mm_add_ps(c1, _mm_loadu_ps(&C[j + i*n])));
        _mm_storeu_ps(&C[j + i*n + 4], _mm_add_ps(c2, _mm_loadu_ps(&C[j + i*n + 4])));
        _mm_storeu_ps(&C[j + i*n + 8], _mm_add_ps(c3, _mm_loadu_ps(&C[j + i*n + 8])));
        _mm_storeu_ps(&C[j + i*n + 12], _mm_add_ps(c4, _mm_loadu_ps(&C[j + i*n + 12])));
        _mm_storeu_ps(&C[j + i*n + 16], _mm_add_ps(c5, _mm_loadu_ps(&C[j + i*n + 16])));
        _mm_storeu_ps(&C[j + i*n + 20], _mm_add_ps(c6, _mm_loadu_ps(&C[j + i*n + 20])));
        _mm_storeu_ps(&C[j + i*n + 24], _mm_add_ps(c7, _mm_loadu_ps(&C[j + i*n + 24])));
        _mm_storeu_ps(&C[j + i*n + 28], _mm_add_ps(c8, _mm_loadu_ps(&C[j + i*n + 28])));

        c1 = _mm_setzero_ps();
        c2 = _mm_setzero_ps();
        c3 = _mm_setzero_ps();
        c4 = _mm_setzero_ps();
        c5 = _mm_setzero_ps();
        c6 = _mm_setzero_ps();
        c7 = _mm_setzero_ps();
        c8 = _mm_setzero_ps();
      }

      //handle fringe cases using sse instructions
      for (int fringe = (n/32)*32; fringe < (n/16)*16; fringe += 16) {
        for (int k = kk; k < xMax; k++) {
          b1 = _mm_load1_ps(&B[k + i*n]);
          a1 = _mm_loadu_ps(&A[fringe + k*n]);
          c1 = _mm_add_ps(c1, _mm_mul_ps(a1, b1));
          a1 = _mm_loadu_ps(&A[fringe + k*n + 4]);
          c2 = _mm_add_ps(c2, _mm_mul_ps(a1, b1));
          a1 = _mm_loadu_ps(&A[fringe + k*n + 8]);
          c3 = _mm_add_ps(c3, _mm_mul_ps(a1, b1));
          a1 = _mm_loadu_ps(&A[fringe + k*n + 12]);
          c4 = _mm_add_ps(c4, _mm_mul_ps(a1, b1));
        }

        _mm_storeu_ps(&C[fringe + i*n], _mm_add_ps(c1, _mm_loadu_ps(&C[fringe + i*n])));
        _mm_storeu_ps(&C[fringe + i*n + 4], _mm_add_ps(c2, _mm_loadu_ps(&C[fringe + i*n + 4])));
        _mm_storeu_ps(&C[fringe + i*n + 8], _mm_add_ps(c3, _mm_loadu_ps(&C[fringe + i*n + 8])));
        _mm_storeu_ps(&C[fringe + i*n + 12], _mm_add_ps(c4, _mm_loadu_ps(&C[fringe + i*n + 12])));

        c1 = _mm_setzero_ps();
        c2 = _mm_setzero_ps();
        c3 = _mm_setzero_ps();
        c4 = _mm_setzero_ps();
      }
      for (int fringe = (n/32)*32; fringe < (n/16)*16; fringe += 16) {
        for (int k = kk; k < xMax; k++) {
          b1 = _mm_load1_ps(&B[k + i*n]);
          a1 = _mm_loadu_ps(&A[fringe + k*n]);
          c1 = _mm_add_ps(c1, _mm_mul_ps(a1, b1));
          a1 = _mm_loadu_ps(&A[fringe + k*n + 4]);
          c2 = _mm_add_ps(c2, _mm_mul_ps(a1, b1));
          a1 = _mm_loadu_ps(&A[fringe + k*n + 8]);
          c3 = _mm_add_ps(c3, _mm_mul_ps(a1, b1));
          a1 = _mm_loadu_ps(&A[fringe + k*n + 12]);
          c4 = _mm_add_ps(c4, _mm_mul_ps(a1, b1));
        }

        _mm_storeu_ps(&C[fringe + i*n], _mm_add_ps(c1, _mm_loadu_ps(&C[fringe + i*n])));
        _mm_storeu_ps(&C[fringe + i*n + 4], _mm_add_ps(c2, _mm_loadu_ps(&C[fringe + i*n + 4])));
        _mm_storeu_ps(&C[fringe + i*n + 8], _mm_add_ps(c3, _mm_loadu_ps(&C[fringe + i*n + 8])));
        _mm_storeu_ps(&C[fringe + i*n + 12], _mm_add_ps(c4, _mm_loadu_ps(&C[fringe + i*n + 12])));

        c1 = _mm_setzero_ps();
        c2 = _mm_setzero_ps();
        c3 = _mm_setzero_ps();
        c4 = _mm_setzero_ps();
      }
      for (int fringe = (n/8)*8; fringe < n; fringe += 1) {
        for (int k = kk; k < xMax; k++) {
          C[fringe + i*n] += B[k + i*n] * A[fringe + k*n];
        }
      }
    }
  }
}