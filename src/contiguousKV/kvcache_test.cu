#include "kvcache.cuh"
#include <iostream>
#include <vector>

using namespace transfomer;

__global__ void fill_half(half* p, int n, half v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

int main() {
    KVCacheConfig cfg;
    cfg.n_layers    = 4;
    cfg.n_kv_heads  = 8;
    cfg.head_dim    = 32;
    cfg.max_seq_len = 16;
    cfg.batch       = 1;

    KVCacheContiguous kv(cfg);

    const int slice_elems = cfg.n_kv_heads * cfg.head_dim;
    half* k_tmp = nullptr;
    half* v_tmp = nullptr;
    cuda_check(cudaMalloc(&k_tmp, slice_elems * sizeof(half)), "cudaMalloc k_tmp");
    cuda_check(cudaMalloc(&v_tmp, slice_elems * sizeof(half)), "cudaMalloc v_tmp");

    // token 0 append for all layers
    for (int layer = 0; layer < cfg.n_layers; ++layer) {
        fill_half<<<(slice_elems+255)/256, 256>>>(k_tmp, slice_elems, __float2half(1.0f + layer));
        fill_half<<<(slice_elems+255)/256, 256>>>(v_tmp, slice_elems, __float2half(10.0f + layer));
        kv.append_layer_current(layer, k_tmp, v_tmp);
    }
    kv.commit_token();

    auto view0 = kv.view_layer(0);
    std::cout << "cur_len=" << view0.cur_len
              << " stride_token_elems=" << view0.stride_token_elems << "\n";

    cudaFree(k_tmp);
    cudaFree(v_tmp);
    std::cout << "OK\n";
    return 0;
}
