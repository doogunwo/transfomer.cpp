#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "ggml.h" 

#define QK_GTF 256

// [수정] GTF는 우리가 float으로 만들었으니 그대로 둠
typedef struct {
    float   d;          
    uint8_t lsb[32];
    uint8_t b1[32];
    uint8_t b2[32];
    uint8_t msb[32];
} block_gtf;

// ★ [수정 핵심] float -> uint16_t 로 변경!
// 실제 라이브러리는 Scale 값을 16비트(FP16)로 저장합니다.
typedef struct {
    uint16_t d;      // float(4byte)가 아니라 fp16(2byte)여야 함!
    int8_t  qs[256];
} block_q8_0;

extern void ggml_vec_dot_gtf_q8_0(const int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);

int main() {
    printf("=== GTF Kernel Test Start ===\n");
    int n = 256;
    block_gtf  x_gtf;
    block_q8_0 y_q8;

    // 1. 스케일 설정
    x_gtf.d = 1.0f;     // GTF는 float이라 1.0f
    y_q8.d  = 0x3C00;   // ★ [수정] FP16에서 1.0은 16진수로 0x3C00 입니다.

    // 2. 초기화
    memset(x_gtf.lsb, 0, 32);
    memset(x_gtf.b1,  0, 32);
    memset(x_gtf.b2,  0, 32);
    memset(x_gtf.msb, 0, 32);
    for(int i=0; i<256; i++) y_q8.qs[i] = 0;

    // 3. 테스트 케이스: 3 * 10 = 30
    x_gtf.lsb[0] |= 1; 
    x_gtf.b1[0]  |= 1;
    y_q8.qs[0] = 10;

    // 4. 실행
    float result = 0.0f;
    ggml_vec_dot_gtf_q8_0(n, &result, 0, &x_gtf, 0, &y_q8, 0, 1);

    printf("Expected: 30.000000\n");
    printf("Actual:   %f\n", result);

    if (result == 30.0f) printf(">>> SUCCESS!\n");
    else printf(">>> FAILED\n");

    return 0;
}