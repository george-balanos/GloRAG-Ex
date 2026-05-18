#include <ATen/native/cuda/gdr_utils.h>
#include <gdrapi.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Logging.h>

#include <cstdio>
#include <cstring>
#include <chrono>
#include <unordered_map>
#include <mutex>

namespace at {
namespace native {

static constexpr size_t kGpuPageSize = 1ULL << 16;
static constexpr size_t kGpuPageMask = ~(kGpuPageSize - 1);

static thread_local bool gdr_enabled = false;

void gdr_enable_for_current_thread()  { gdr_enabled = true; }
void gdr_disable_for_current_thread() { gdr_enabled = false; }
bool gdr_is_enabled() { return gdr_enabled; }



static gdr_t g_gdr = nullptr;
static std::once_flag g_gdr_init;

static gdr_t get_gdr_handle() {
    // printf(">>> GDR HANDLE INITIALIZING <<<\n");
    std::call_once(g_gdr_init, []() {
        g_gdr = gdr_open();
        if (!g_gdr) {
            printf("[GDR] FATAL: gdr_open() failed at init\n");
        }
    });
    return g_gdr;
}

struct GdrCacheEntry {
    gdr_mh_t mh;
    void* mapped;
    size_t pin_size;
    uintptr_t mapping_offset;
    bool has_thread_pool = false;
};

static std::unordered_map<uintptr_t, GdrCacheEntry> gdr_cache;
static std::mutex gdr_cache_mutex;
static size_t gdr_cache_hits = 0;
static size_t gdr_cache_misses = 0;

static void safe_delete_thread_pool(const GdrCacheEntry& e) {
    if (e.has_thread_pool) {
        gdr_delete_thread_pool(e.mh);
    }
}

struct GdrCacheCleanup {
    ~GdrCacheCleanup() {
        std::lock_guard<std::mutex> lock(gdr_cache_mutex);
        if (!g_gdr) return;

        for (auto& [k, v] : gdr_cache) {
            safe_delete_thread_pool(v);
            gdr_unmap(g_gdr, v.mh, v.mapped, v.pin_size);
            gdr_unpin_buffer(g_gdr, v.mh);
        }

        gdr_close(g_gdr);
        g_gdr = nullptr;

        printf("[GDR CACHE] cleanup done — hits=%zu misses=%zu\n",
               gdr_cache_hits, gdr_cache_misses);
    }
};
static GdrCacheCleanup gdr_cleanup;

void gdr_evict_cached_region(void* gpu_ptr) {
    uintptr_t aligned = (uintptr_t)gpu_ptr & kGpuPageMask;

    std::lock_guard<std::mutex> lock(gdr_cache_mutex);
    auto it = gdr_cache.find(aligned);
    if (it == gdr_cache.end()) return;

    if (g_gdr) {
        safe_delete_thread_pool(it->second);
        gdr_unmap(g_gdr, it->second.mh, it->second.mapped, it->second.pin_size);
        gdr_unpin_buffer(g_gdr, it->second.mh);
    }

    gdr_cache.erase(it);
}

size_t gdr_copy_size_threshold() {
    // printf("[GDR] Using copy size threshold of %f bytes\n", 16.0f);
    return 4 * 1024;
}

bool gdr_copy_d2h(void* dst, void* gpu_ptr, size_t size) {

    static size_t below_threshold_count = 0;
    static size_t above_threshold_count = 0;

    if (size > gdr_copy_size_threshold()) {
        above_threshold_count++;
        return false;
    }
    below_threshold_count++;

    gdr_t g = get_gdr_handle();
    if (!g) return false;

    uintptr_t aligned = (uintptr_t)gpu_ptr & kGpuPageMask;
    size_t offset = (uintptr_t)gpu_ptr - aligned;
    size_t pin_size = (size + offset + kGpuPageSize - 1) & kGpuPageMask;

    // auto t_lookup_start = std::chrono::high_resolution_clock::now();

    GdrCacheEntry entry;
    bool cache_hit = false;

    {
        std::lock_guard<std::mutex> lock(gdr_cache_mutex);
        auto it = gdr_cache.find(aligned);

        if (it != gdr_cache.end()) {

            if (pin_size <= it->second.pin_size) {
                entry = it->second;
                cache_hit = true;
                gdr_cache_hits++;
            } else {
                safe_delete_thread_pool(it->second);

                gdr_unmap(g, it->second.mh, it->second.mapped, it->second.pin_size);
                gdr_unpin_buffer(g, it->second.mh);

                gdr_cache.erase(it);
                gdr_cache_misses++;
            }
        } else {
            gdr_cache_misses++;
        }
    }

    // auto t_lookup_end = std::chrono::high_resolution_clock::now();

    if (!cache_hit) {

        gdr_mh_t mh;

        if (gdr_pin_buffer(g, (CUdeviceptr)aligned, pin_size, 0, 0, &mh) != 0) {
            printf("[GDR] gdr_pin_buffer() failed\n");
            return false;
        }

        void* mapped = nullptr;
        if (gdr_map(g, mh, &mapped, pin_size) != 0) {
            printf("[GDR] gdr_map() failed\n");
            gdr_unpin_buffer(g, mh);
            return false;
        }

        int threads = 4;
        bool tp_ok = (gdr_create_thread_pool(mh, threads) == 0);

        gdr_info_t info;
        if (gdr_get_info(g, mh, &info) != 0) {
            printf("[GDR] gdr_get_info() failed\n");

            if (tp_ok) gdr_delete_thread_pool(mh);

            gdr_unmap(g, mh, mapped, pin_size);
            gdr_unpin_buffer(g, mh);
            return false;
        }

        uintptr_t mapping_offset = (uintptr_t)info.va & (kGpuPageSize - 1);

        entry = { mh, mapped, pin_size, mapping_offset, tp_ok };

        {
            std::lock_guard<std::mutex> lock(gdr_cache_mutex);
            gdr_cache[aligned] = entry;
        }
    }

    void* src = (char*)entry.mapped + entry.mapping_offset + offset;

    // auto t_copy_start = std::chrono::high_resolution_clock::now();

    gdr_copy_from_mapping_optimized(
        entry.mh,
        dst,
        src,
        (CUdeviceptr)gpu_ptr,
        size
    );

    // auto t_copy_end = std::chrono::high_resolution_clock::now();

    // auto us = [](auto a, auto b) {
    //     return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
    // };

    // printf("[GDR] size=%6zu | hit=%-3s | lookup=%4ldus | copy=%6ldus | hits=%zu misses=%zu | above=%zu below=%zu\n",
    //     size,
    //     cache_hit ? "YES" : "NO",
    //     us(t_lookup_start, t_lookup_end),
    //     us(t_copy_start, t_copy_end),
    //     gdr_cache_hits,
    //     gdr_cache_misses,
    //     above_threshold_count,
    //     below_threshold_count);

    return true;
}

} // namespace native
} // namespace at