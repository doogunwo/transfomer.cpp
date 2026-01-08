#pragma once

#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace transfomer {

// -----------------------------
// CUDA error helper
// -----------------------------
inline void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("[CUDA] ") + what + ": " + cudaGetErrorString(e));
    }
}

// -----------------------------
// RAII device buffer (cudaMalloc/cudaFree)
// -----------------------------
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t bytes) { allocate(bytes); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept { *this = std::move(other); }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this == &other) return *this;
        reset();
        ptr_   = other.ptr_;
        bytes_ = other.bytes_;
        other.ptr_ = nullptr;
        other.bytes_ = 0;
        return *this;
    }

    ~DeviceBuffer() { reset(); }

    void allocate(size_t bytes) {
        reset();
        if (bytes == 0) return;
        void* p = nullptr;
        cuda_check(cudaMalloc(&p, bytes), "cudaMalloc");
        ptr_ = p;
        bytes_ = bytes;
    }

    void reset() {
        if (ptr_) {
            cudaFree(ptr_); // best-effort in dtor
        }
        ptr_ = nullptr;
        bytes_ = 0;
    }

    void* data() { return ptr_; }
    const void* data() const { return ptr_; }
    size_t size_bytes() const { return bytes_; }
    explicit operator bool() const { return ptr_ != nullptr; }

private:
    void*  ptr_   = nullptr;
    size_t bytes_ = 0;
};

// -----------------------------
// Contiguous KV cache config (batch=1 first)
// layout (logical):
//   K[layer][t][kv_head][head_dim]  (half)
//   V[layer][t][kv_head][head_dim]  (half)
//
// underlying linear index:
//   idx = (((layer * B + b) * S + t) * H + h) * D + d
// -----------------------------
struct KVCacheConfig {
    int n_layers    = 0;
    int n_kv_heads  = 0;
    int head_dim    = 0;
    int max_seq_len = 0;
    int batch       = 1;  // start with 1
};

struct KVView {
    // base pointers for this layer at token 0 (prefix view)
    const half* k_base = nullptr;
    const half* v_base = nullptr;

    // current valid prefix length [0, cur_len)
    int cur_len = 0;

    // stride in elements
    size_t stride_token_elems = 0; // H*D
    size_t stride_layer_elems = 0; // B*S*H*D (for completeness)

    int n_kv_heads = 0;
    int head_dim   = 0;
};

// -----------------------------
// Contiguous KV cache (GPU)
// - owns K/V device buffers
// - append per-layer per-token (device-to-device copy)
// - view returns prefix pointer + metadata
// -----------------------------
class KVCacheContiguous {
public:
    KVCacheContiguous() = default;
    explicit KVCacheContiguous(KVCacheConfig cfg) { init(std::move(cfg)); }

    KVCacheContiguous(const KVCacheContiguous&) = delete;
    KVCacheContiguous& operator=(const KVCacheContiguous&) = delete;

    KVCacheContiguous(KVCacheContiguous&&) noexcept = default;
    KVCacheContiguous& operator=(KVCacheContiguous&&) noexcept = default;

    void init(KVCacheConfig cfg) {
        if (cfg.n_layers <= 0 || cfg.n_kv_heads <= 0 || cfg.head_dim <= 0 || cfg.max_seq_len <= 0) {
            throw std::invalid_argument("KVCacheConfig: invalid dimensions");
        }
        if (cfg.batch <= 0) {
            throw std::invalid_argument("KVCacheConfig: batch must be >= 1");
        }
        cfg_ = cfg;
        cur_len_ = 0;

        const size_t total_elems = static_cast<size_t>(cfg_.n_layers)
                                 * static_cast<size_t>(cfg_.batch)
                                 * static_cast<size_t>(cfg_.max_seq_len)
                                 * static_cast<size_t>(cfg_.n_kv_heads)
                                 * static_cast<size_t>(cfg_.head_dim);

        const size_t total_bytes = total_elems * sizeof(half);

        k_.allocate(total_bytes);
        v_.allocate(total_bytes);

        // optional: zero init (debug friendly). can remove later for speed.
        cuda_check(cudaMemset(k_.data(), 0, total_bytes), "cudaMemset(K)");
        cuda_check(cudaMemset(v_.data(), 0, total_bytes), "cudaMemset(V)");
    }

    const KVCacheConfig& config() const { return cfg_; }

    int cur_len() const { return cur_len_; }
    int max_seq_len() const { return cfg_.max_seq_len; }

    void reset_len(int new_len = 0) {
        if (new_len < 0 || new_len > cfg_.max_seq_len) {
            throw std::out_of_range("reset_len: out of range");
        }
        cur_len_ = new_len;
    }

    // advance after finishing all layers for the new token
    void commit_token() {
        if (cur_len_ >= cfg_.max_seq_len) {
            throw std::out_of_range("commit_token: max_seq_len reached");
        }
        cur_len_ += 1;
    }

    // Compute destination pointer for K/V at (layer, b, t)
    // token slice shape: [n_kv_heads * head_dim] (half)
    half* k_token_ptr(int layer, int b, int t) {
        return reinterpret_cast<half*>(k_.data()) + offset_elems(layer, b, t);
    }
    half* v_token_ptr(int layer, int b, int t) {
        return reinterpret_cast<half*>(v_.data()) + offset_elems(layer, b, t);
    }

    const half* k_token_ptr(int layer, int b, int t) const {
        return reinterpret_cast<const half*>(k_.data()) + offset_elems(layer, b, t);
    }
    const half* v_token_ptr(int layer, int b, int t) const {
        return reinterpret_cast<const half*>(v_.data()) + offset_elems(layer, b, t);
    }

    // Append one token's KV for a given layer.
    // - src pointers must be device pointers (half*), contiguous length H*D
    // - token_idx is where to write (usually cur_len()).
    void append_layer_token(
        int layer,
        int token_idx,
        const half* k_src_dev,
        const half* v_src_dev,
        cudaStream_t stream = 0,
        int b = 0
    ) {
        bounds_check(layer, b, token_idx);

        const size_t slice_elems = static_cast<size_t>(cfg_.n_kv_heads) * static_cast<size_t>(cfg_.head_dim);
        const size_t slice_bytes = slice_elems * sizeof(half);

        half* k_dst = k_token_ptr(layer, b, token_idx);
        half* v_dst = v_token_ptr(layer, b, token_idx);

        cuda_check(cudaMemcpyAsync(k_dst, k_src_dev, slice_bytes, cudaMemcpyDeviceToDevice, stream),
                   "cudaMemcpyAsync(K slice)");
        cuda_check(cudaMemcpyAsync(v_dst, v_src_dev, slice_bytes, cudaMemcpyDeviceToDevice, stream),
                   "cudaMemcpyAsync(V slice)");
    }

    // Convenience: append at current cur_len (decode step), per layer
    void append_layer_current(
        int layer,
        const half* k_src_dev,
        const half* v_src_dev,
        cudaStream_t stream = 0,
        int b = 0
    ) {
        append_layer_token(layer, cur_len_, k_src_dev, v_src_dev, stream, b);
    }

    // Prefix view for a layer (base pointers at token 0) + current valid length
    KVView view_layer(int layer, int b = 0) const {
        if (layer < 0 || layer >= cfg_.n_layers) throw std::out_of_range("view_layer: layer");
        if (b < 0 || b >= cfg_.batch) throw std::out_of_range("view_layer: batch index");

        KVView v;
        v.k_base = k_token_ptr(layer, b, 0);
        v.v_base = v_token_ptr(layer, b, 0);
        v.cur_len = cur_len_;
        v.stride_token_elems = static_cast<size_t>(cfg_.n_kv_heads) * static_cast<size_t>(cfg_.head_dim);
        v.stride_layer_elems = static_cast<size_t>(cfg_.batch)
                             * static_cast<size_t>(cfg_.max_seq_len)
                             * v.stride_token_elems;
        v.n_kv_heads = cfg_.n_kv_heads;
        v.head_dim   = cfg_.head_dim;
        return v;
    }

    // Raw buffers (for debugging / advanced kernels)
    const half* k_data() const { return reinterpret_cast<const half*>(k_.data()); }
    const half* v_data() const { return reinterpret_cast<const half*>(v_.data()); }
    half* k_data() { return reinterpret_cast<half*>(k_.data()); }
    half* v_data() { return reinterpret_cast<half*>(v_.data()); }

private:
    size_t offset_elems(int layer, int b, int t) const {
        // idx = (((layer * B + b) * S + t) * H) * D
        const size_t B = static_cast<size_t>(cfg_.batch);
        const size_t S = static_cast<size_t>(cfg_.max_seq_len);
        const size_t H = static_cast<size_t>(cfg_.n_kv_heads);
        const size_t D = static_cast<size_t>(cfg_.head_dim);

        const size_t lb = static_cast<size_t>(layer) * B + static_cast<size_t>(b);
        return ((lb * S + static_cast<size_t>(t)) * H) * D;
    }

    void bounds_check(int layer, int b, int t) const {
        if (layer < 0 || layer >= cfg_.n_layers) throw std::out_of_range("append: layer");
        if (b < 0 || b >= cfg_.batch) throw std::out_of_range("append: batch index");
        if (t < 0 || t >= cfg_.max_seq_len) throw std::out_of_range("append: token_idx");
    }

    KVCacheConfig cfg_{};
    int cur_len_ = 0;

    DeviceBuffer k_;
    DeviceBuffer v_;
};

} // namespace transfomer
