#pragma once

// #include "llama.h"          // TODO: enable when you have the C API header
// #include "llama-impl.h"     // TODO: port/implement then enable
// #include "llama-arch.h"     // TODO: port/implement then enable
// #include "llama-mmap.h"     // TODO: port/implement then enable
// #include "ggml-cpp.h"       // TODO: port/implement then enable

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>        // sscanf
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

struct ggml_tensor;
struct ggml_context;
struct gguf_context;

using ggml_backend_buffer_t = void*;
//TODO: replace with real ggml-backnd-buffer-t

class llama_file {
public:
    [[nodiscard]] std::size_t size() const noexcept;
};

//TODO
class llama_mlocks;
class llama_mmaps;
class llama_files;
class llama_model_kv_override;
class llama_model_tensor_buft_override;

enum llama_ftype    :   int {LLAMA_FTYPE_UNKNOWN = 0};
enum llm_kv         :   int;
enum llm_arch       :   int {LLM_ARCH_UNKNOWN = 0};

using llama_progress_callback = bool(*)(float progress, void* user_data);

inline void gguf_ctx_noop_deleter(gguf_context*) {}
inline void ggml_ctx_noop_deleter(ggml_context*) {}

using gguf_context_ptr = std::unique_ptr<gguf_context, decltype(&gguf_ctx_noop_deleter)>;
using ggml_context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_ctx_noop_deleter)>;


class LLM_KV{
    public:
        explicit LLM_KV(llm_arch) {}
};

inline std::string format(const char*, ...) {
    return "format() not implemented"; // fixed
}

using llama_buf_map = std::unordered_map<std::uint32_t, ggml_backend_buffer_t>;

// -------------
// File version enm
// -------------
enum class llama_fver : int {
    GGUF_FILE_VERSION_V1 = 1,
    GGUF_FILE_VERSION_V2 = 2,
    GGUF_FILE_VERSION_V3 = 3,
};

const char* llama_file_version_name(llama_fver version);

// -------------
// llama model loader 
// -------------
class LlamaModelLoader final {

    private:
    // ---- counters ----
    int n_kv_      = 0;
    int n_tensors_ = 0;
    int n_created_ = 0;

    std::uint64_t n_elements_ = 0;
    std::size_t   n_bytes_    = 0;

    bool use_mmap_       = false;
    bool check_tensors_  = false;
    bool no_alloc_       = false;

    public:
        static constexpr int TENSOR_NOT_REQUIRED    = 1 << 0;
        static constexpr int TENSOR_DUPLICATED      = 1 << 1;
        static constexpr int TENSOR_SKIP            = 1 << 2;

        LlamaModelLoader(
            const std::string& fname,
            std::vector<std::string>& splits, // optional, only needed if split does not follow naming scheme
            bool use_mmap,
            bool check_tensors,
            bool no_alloc,
            const llama_model_kv_override* param_overrides_p,
            const llama_model_tensor_buft_override* param_tensor_buft_overrides_p
        );

        ~LlamaModelLoader() = default;
        LlamaModelLoader(const LlamaModelLoader&)               = delete;   
        LlamaModelLoader& operator=(const LlamaModelLoader&)    = delete;
        LlamaModelLoader(LlamaModelLoader&&)                    = default;
        LlamaModelLoader& operator=(LlamaModelLoader&&)          = default; 

        [[nodiscard]] int kv_count() const noexcept { return n_kv_; }
        [[nodiscard]] int tensor_count() const noexcept { return n_tensors_; }
        [[nodiscard]] int created_count() const noexcept { return n_created_; }

        [[nodiscard]] std::uint64_t element_count() const noexcept { return n_elements_; }
        [[nodiscard]] std::size_t   byte_count()    const noexcept { return n_bytes_; }

        [[nodiscard]] bool uses_mmap() const noexcept { return use_mmap_; }
        [[nodiscard]] bool checks_tensors() const noexcept { return check_tensors_; }
        [[nodiscard]] bool no_alloc_mode() const noexcept { return no_alloc_; }

        // ===== KV getters =====
        template <typename T>
        typename std::enable_if<std::is_integral<T>::value, bool>::type
        get_arr_n(const std::string& key, T& result, bool required = true);

        template <typename T>
        typename std::enable_if<std::is_integral<T>::value, bool>::type
        get_arr_n(enum llm_kv kid, T& result, bool required = true);

        template <typename T>
        bool get_arr(const std::string& key, std::vector<T>& result, bool required = true);

        template <typename T, std::size_t N_MAX>
        bool get_arr(const std::string& key, std::array<T, N_MAX>& result, bool required = true);

        template <typename T>
        bool get_arr(enum llm_kv kid, T& result, bool required = true);

        template <typename T>
        bool get_key(const std::string& key, T& result, bool required = true);

        template <typename T>
        bool get_key(enum llm_kv kid, T& result, bool required = true);

        template <typename T, std::size_t N_MAX>
        bool get_key_or_arr(const std::string& key, std::array<T, N_MAX>& result, std::uint32_t n, bool required = true);

        template <typename T>
        bool get_key_or_arr(enum llm_kv kid, T& result, std::uint32_t n, bool required = true);

        bool get_key_or_arr(enum llm_kv kid, std::uint32_t& result, bool required = true);

};
