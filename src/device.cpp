// device.cpp - CPU-only stubs for prima.cpp device functions
#include "profiler.h"
#include "llama.h"
#include <cstring>
#include <string>
#include <fstream>

// Device detection stubs (CPU-only)
int  device_has_cuda()    { return 0; }
int  device_has_blas()    { return 0; }
int  device_has_sycl()    { return 0; }
int  device_has_metal()   { return 0; }
int  device_has_vulkan()  { return 0; }
int  device_has_kompute() { return 0; }
int  device_has_gpublas() { return 0; }

const char* device_name() { return "CPU"; }
const char* device_os()   { return "Linux"; }

uint32_t device_cpu_cores() { return 2; }

uint64_t device_physical_memory(bool available) {
    std::ifstream f("/proc/meminfo");
    std::string line;
    uint64_t total = 0, avail = 0;
    while (std::getline(f, line)) {
        if (line.find("MemTotal:") == 0)
            sscanf(line.c_str(), "MemTotal: %lu", &total);
        if (line.find("MemAvailable:") == 0)
            sscanf(line.c_str(), "MemAvailable: %lu", &avail);
    }
    return (available ? avail : total) * 1024;
}

uint64_t device_swappable_memory() { return 0; }
uint64_t device_swap_memory(bool available) { return 0; }

float device_memory_bw(int n_threads) { return 10.0f; }
float device_cpu_mem_copy(struct llama_model* model, int n_threads) { return 0.001f; }

void device_disk_seq_bw(float* read_bw, float* write_bw, int n_threads) {
    *read_bw = 500.0f; *write_bw = 400.0f;
}

void device_disk_rnd_bw(float* read_bw, float* write_bw, int n_threads) {
    *read_bw = 50.0f; *write_bw = 40.0f;
}

float device_inp_embd_delay(struct llama_model* model, enum ggml_type src0t, int n_tokens, int n_threads) {
    return 0.001f;
}

void device_get_props(struct llama_model* model, int dev, struct ggml_backend_dev_props* props) {}

float device_cpu_flops(struct llama_model* model, enum ggml_type src0t, enum ggml_type src1t, int n_threads) { return 1.0f; }
float device_cuda_flops(struct llama_model* model, enum ggml_type src0t, enum ggml_type src1t) { return 0.0f; }
float device_metal_flops(struct llama_model* model, enum ggml_type src0t, enum ggml_type src1t) { return 0.0f; }

size_t serialize(const struct device_info* dev_info, char** buf) { if (buf) *buf = nullptr; return 0; }
void deserialize(const char* buf, struct device_info* dev_info) {}

void device_print_props(struct device_info* dev_info_set, int n, struct llama_model* model, const struct llama_context_params cparams) {}
