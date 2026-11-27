#include "ggml-xdna.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_xclbin.h>

#include <algorithm>
#include <charconv>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

struct matmul_shape {
    int64_t m;
    int64_t k;
    int64_t n;
    const char * xclbin;
    const char * instructions;
};

// IRON whole-array kernels use BF16 inputs, column-major B, and row-major F32 C.
constexpr matmul_shape standard_shape = { 256, 1024, 1024, "matmul.xclbin", "insts.bin" };
constexpr matmul_shape small_shape    = { 128, 1024, 1024, "matmul-small.xclbin", "insts-small.bin" };
constexpr matmul_shape wide_shape     = { 256, 4096, 1024, "matmul-wide.xclbin", "insts-wide.bin" };
constexpr matmul_shape output_shape   = { 256, 1024, 4096, "matmul-output.xclbin", "insts-output.bin" };
constexpr size_t input_bo_budget = 4 * 1024 * 1024;
constexpr size_t max_instruction_bytes = 1024 * 1024;

constexpr const char * device_name = "XDNA";
constexpr const char * device_description = "AMD XDNA NPU (XRT)";
ggml_guid backend_guid = { 0x1e, 0x3c, 0x11, 0xb7, 0x05, 0x90, 0x42, 0x27, 0xa5, 0x52, 0xbd, 0x54, 0xf5, 0x9b, 0xf4, 0xde };

std::vector<uint32_t> read_instructions(const fs::path & file) {
    std::ifstream input(file, std::ios::binary | std::ios::ate);
    const auto bytes = static_cast<std::streamoff>(input.tellg());
    if (bytes <= 0 || bytes > static_cast<std::streamoff>(max_instruction_bytes) || bytes % sizeof(uint32_t) != 0) {
        throw std::runtime_error("invalid XDNA instruction file");
    }
    std::vector<uint32_t> instructions(static_cast<size_t>(bytes) / sizeof(uint32_t));
    input.seekg(0);
    if (!input.read(reinterpret_cast<char *>(instructions.data()), bytes)) {
        throw std::runtime_error("failed to read XDNA instruction file");
    }
    return instructions;
}

xrt::xclbin load_xclbin(const fs::path & file) {
    const auto utf8 = file.u8string();
    return xrt::xclbin(std::string(utf8.begin(), utf8.end()));
}

bool is_layout_op(ggml_op op) {
    return op == GGML_OP_NONE || op == GGML_OP_VIEW || op == GGML_OP_RESHAPE ||
           op == GGML_OP_TRANSPOSE || op == GGML_OP_PERMUTE;
}

unsigned env_unsigned(const char * name, unsigned fallback) {
    const char * value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return fallback;
    }
    const char * end = value + std::strlen(value);
    unsigned result = 0;
    const auto parsed = std::from_chars(value, end, result);
    if (parsed.ec == std::errc{} && parsed.ptr == end) {
        return result;
    }
    GGML_LOG_WARN("ggml-xdna: invalid %s, using %u\n", name, fallback);
    return fallback;
}

bool is_host_tensor(const ggml_tensor * tensor) {
    auto * buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    return buffer == nullptr || ggml_backend_buffer_is_host(buffer);
}

bool supports_matmul(const ggml_tensor * op) {
    if (op->op != GGML_OP_MUL_MAT || op->type != GGML_TYPE_F32 || ggml_is_empty(op) ||
        ggml_prec_get_acc(op) != GGML_PREC_UNDEFINED || ggml_prec_get_src(op, 1) == GGML_PREC_F32) {
        return false;
    }
    const ggml_tensor * weights = op->src[0];
    const ggml_tensor * input = op->src[1];
    if (weights == nullptr || input == nullptr || input->type != GGML_TYPE_F32) {
        return false;
    }
    const auto * traits = ggml_get_type_traits(weights->type);
    static const int64_t min_rows = std::max(1u, env_unsigned("GGML_XDNA_MIN_ROWS", 256));
    return (weights->type == GGML_TYPE_F32 || traits->to_float != nullptr) &&
           traits->blck_size > 0 && standard_shape.k % traits->blck_size == 0 &&
           is_host_tensor(weights) && is_host_tensor(input) && is_host_tensor(op) &&
           ggml_is_contiguous_rows(weights) && ggml_is_contiguous_rows(input) && ggml_is_contiguous(op) &&
           weights->ne[0] == input->ne[0] && input->ne[0] >= 256 &&
           weights->ne[0] % traits->blck_size == 0 && weights->ne[1] >= 128 && input->ne[1] >= min_rows &&
           op->ne[0] == weights->ne[1] && op->ne[1] == input->ne[1] &&
           op->ne[2] == input->ne[2] && op->ne[3] == input->ne[3] &&
           weights->ne[2] > 0 && weights->ne[3] > 0 &&
           input->ne[2] % weights->ne[2] == 0 && input->ne[3] % weights->ne[3] == 0;
}

unsigned device_index() {
    static const unsigned index = env_unsigned("GGML_XDNA_DEVICE_INDEX", 0);
    return index;
}

bool kernel_files_exist(const fs::path & dir, const matmul_shape & shape) {
    return fs::is_regular_file(dir / shape.xclbin) && fs::is_regular_file(dir / shape.instructions);
}

fs::path find_kernel_directory() {
    const char * value = std::getenv("GGML_XDNA_KERNEL_DIR");
    if (value == nullptr || *value == '\0') {
        return {};
    }
    try {
        const auto dir = fs::absolute(fs::u8path(value));
        if (!kernel_files_exist(dir, standard_shape)) {
            throw std::runtime_error("expected matmul.xclbin and insts.bin in GGML_XDNA_KERNEL_DIR");
        }
        // Opening the device verifies the XRT userspace and driver are present.
        const auto device = xrt::device(device_index());
        return dir;
    } catch (const std::exception & error) {
        GGML_LOG_WARN("ggml-xdna: %s\n", error.what());
        return {};
    }
}

const fs::path & kernel_directory() {
    static const auto dir = find_kernel_directory();
    return dir;
}

class xdna_kernel {
public:
    xdna_kernel(xrt::device & device, const fs::path & dir, const matmul_shape & tile)
        : shape(tile),
          context(device, device.register_xclbin(load_xclbin(dir / shape.xclbin))),
          kernel(context, "MLIR_AIE"),
          weights_bo(context, shape.k * shape.n * sizeof(ggml_bf16_t), xrt::bo::flags::host_only, kernel.group_id(4)),
          output_bo(context, shape.m * shape.n * sizeof(float), xrt::bo::flags::host_only, kernel.group_id(5)),
          weights_data(weights_bo.map<ggml_bf16_t *>()),
          output_data(output_bo.map<float *>()),
          run(kernel),
          weight_row(shape.k) {
        const auto instructions = read_instructions(dir / shape.instructions);
        inst_bo = xrt::bo(context, instructions.size() * sizeof(uint32_t), xrt::bo::flags::cacheable, kernel.group_id(1));
        inst_bo.write(instructions.data());
        inst_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        run.set_arg(0, uint64_t{3});
        run.set_arg(1, inst_bo);
        run.set_arg(2, static_cast<uint32_t>(instructions.size()));
        run.set_arg(4, weights_bo);
        run.set_arg(5, output_bo);
        ensure_input_tiles(1);
        verify_layout();
    }

    uint64_t compute(ggml_tensor * dst) {
        const ggml_tensor * weights = dst->src[0];
        const ggml_tensor * input = dst->src[1];
        const int64_t replicas2 = input->ne[2] / weights->ne[2];
        const int64_t replicas3 = input->ne[3] / weights->ne[3];
        uint64_t count = 0;
        for (int64_t i3 = 0; i3 < input->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < input->ne[2]; ++i2) {
                const auto * w = static_cast<const char *>(weights->data) +
                    (i3 / replicas3) * weights->nb[3] + (i2 / replicas2) * weights->nb[2];
                const auto * a = static_cast<const char *>(input->data) + i3 * input->nb[3] + i2 * input->nb[2];
                auto * c = static_cast<char *>(dst->data) + i3 * dst->nb[3] + i2 * dst->nb[2];
                count += compute_matrix(weights, input, w, a, reinterpret_cast<float *>(c));
            }
        }
        return count;
    }

    const matmul_shape shape;

private:
    void ensure_input_tiles(size_t tiles) {
        while (input_bos.size() < tiles) {
            input_bos.emplace_back(context, shape.m * shape.k * sizeof(ggml_bf16_t),
                                   xrt::bo::flags::host_only, kernel.group_id(3));
        }
    }

    void pack_input(xrt::bo & bo, const char * input, size_t stride, int64_t rows, int64_t reduction) {
        auto * data = bo.map<ggml_bf16_t *>();
        if (reduction == shape.k && stride == static_cast<size_t>(shape.k) * sizeof(float)) {
            ggml_fp32_to_bf16_row(reinterpret_cast<const float *>(input), data, rows * shape.k);
        } else {
            for (int64_t i = 0; i < rows; ++i) {
                auto * row = data + i * shape.k;
                ggml_fp32_to_bf16_row(reinterpret_cast<const float *>(input + i * stride), row, reduction);
                std::fill(row + reduction, row + shape.k, ggml_bf16_t{});
            }
        }
        std::fill(data + rows * shape.k, data + shape.m * shape.k, ggml_bf16_t{});
        bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    void pack_weights(const ggml_tensor * weights, const char * data, int64_t rows, int64_t reduction) {
        const auto to_float = ggml_get_type_traits(weights->type)->to_float;
        for (int64_t i = 0; i < rows; ++i) {
            const auto * src = data + i * weights->nb[1];
            auto * dest = weights_data + i * shape.k;
            if (weights->type == GGML_TYPE_BF16) {
                std::memcpy(dest, src, reduction * sizeof(ggml_bf16_t));
            } else if (weights->type == GGML_TYPE_F32) {
                ggml_fp32_to_bf16_row(reinterpret_cast<const float *>(src), dest, reduction);
            } else {
                to_float(src, weight_row.data(), reduction);
                ggml_fp32_to_bf16_row(weight_row.data(), dest, reduction);
            }
            std::fill(dest + reduction, dest + shape.k, ggml_bf16_t{});
        }
        std::fill(weights_data + rows * shape.k, weights_data + shape.n * shape.k, ggml_bf16_t{});
        weights_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    uint64_t compute_matrix(const ggml_tensor * weights, const ggml_tensor * input, const char * w, const char * a, float * c) {
        const int64_t m = input->ne[1];
        const int64_t k = input->ne[0];
        const int64_t n = weights->ne[1];
        const int64_t available_tiles = input_bo_budget / (shape.m * shape.k * sizeof(ggml_bf16_t));
        const int64_t row_tiles = std::min(available_tiles, (m - 1) / shape.m + 1);
        const int64_t row_window = row_tiles * shape.m;
        ensure_input_tiles(static_cast<size_t>(row_tiles));
        uint64_t count = 0;

        for (int64_t first = 0; first < m; first += row_window) {
            const int64_t end = first + std::min(row_window, m - first);
            for (int64_t kk = 0; kk < k; kk += shape.k) {
                const int64_t reduction = std::min(shape.k, k - kk);
                const size_t w_offset = ggml_row_size(weights->type, kk);
                for (int64_t i = first; i < end; i += shape.m) {
                    pack_input(input_bos[(i - first) / shape.m], a + i * input->nb[1] + kk * sizeof(float),
                               input->nb[1], std::min(shape.m, end - i), reduction);
                }
                for (int64_t j = 0; j < n; j += shape.n) {
                    const int64_t cols = std::min(shape.n, n - j);
                    pack_weights(weights, w + j * weights->nb[1] + w_offset, cols, reduction);
                    for (int64_t i = first; i < end; i += shape.m) {
                        dispatch(input_bos[(i - first) / shape.m]);
                        ++count;
                        const int64_t rows = std::min(shape.m, end - i);
                        for (int64_t row = 0; row < rows; ++row) {
                            const auto * tile = output_data + row * shape.n;
                            auto * dest = c + (i + row) * n + j;
                            if (kk == 0) {
                                std::copy_n(tile, cols, dest);
                            } else {
                                for (int64_t col = 0; col < cols; ++col) {
                                    dest[col] += tile[col];
                                }
                            }
                        }
                    }
                }
            }
        }
        return count;
    }

    void dispatch(const xrt::bo & input) {
        run.set_arg(3, input);
        run.start();
        if (run.wait() != ERT_CMD_STATE_COMPLETED) {
            throw std::runtime_error("XDNA kernel did not complete");
        }
        output_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    }

    void verify_layout() {
        auto & bo = input_bos.front();
        auto * input_data = bo.map<ggml_bf16_t *>();
        std::fill_n(input_data, shape.m * shape.k, ggml_bf16_t{});
        std::fill_n(weights_data, shape.n * shape.k, ggml_bf16_t{});
        input_data[0] = ggml_fp32_to_bf16(2.0f);
        input_data[shape.k + 1] = ggml_fp32_to_bf16(-1.0f);
        input_data[shape.m * shape.k - 1] = ggml_fp32_to_bf16(2.0f);
        weights_data[0] = ggml_fp32_to_bf16(3.0f);
        weights_data[shape.k + 1] = ggml_fp32_to_bf16(4.0f);
        weights_data[shape.n * shape.k - 1] = ggml_fp32_to_bf16(5.0f);
        bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        weights_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        dispatch(bo);
        if (output_data[0] != 6.0f || output_data[1] != 0.0f ||
            output_data[shape.n] != 0.0f || output_data[shape.n + 1] != -4.0f ||
            output_data[shape.m * shape.n - 1] != 10.0f) {
            throw std::runtime_error("XDNA matmul kernel did not pass its layout check");
        }
    }

    xrt::hw_context context;
    xrt::kernel kernel;
    xrt::bo inst_bo;
    std::vector<xrt::bo> input_bos;
    xrt::bo weights_bo;
    xrt::bo output_bo;
    ggml_bf16_t * weights_data;
    float * output_data;
    xrt::run run;
    std::vector<float> weight_row;
};

class xdna_matmul {
public:
    explicit xdna_matmul(const fs::path & dir)
        : directory(dir), device(device_index()), standard(device, directory, standard_shape) {}

    ~xdna_matmul() {
        if (operations != 0) {
            GGML_LOG_INFO("ggml-xdna: executed %" PRIu64 " MUL_MAT operations (%" PRIu64
                          " single-row) in %" PRIu64 " NPU dispatches\n", operations, single_row_ops, dispatches);
        }
    }

    void compute(ggml_tensor * dst) {
        const int64_t m = dst->src[1]->ne[1];
        const int64_t k = dst->src[1]->ne[0];
        const int64_t n = dst->ne[0];
        auto & selected = select_kernel(m, k, n);
        dispatches += selected.compute(dst);
        single_row_ops += m == 1;
        if (++operations == 1) {
            GGML_LOG_INFO("ggml-xdna: first MUL_MAT executed on NPU (M=%" PRId64 ", K=%" PRId64
                          ", N=%" PRId64 ", kernel=%s)\n", m, k, n, selected.shape.xclbin);
        }
    }

    std::mutex mutex;

private:
    struct optional_kernel {
        const matmul_shape & shape;
        std::unique_ptr<xdna_kernel> kernel = nullptr;
        bool checked = false;
    };

    xdna_kernel * load_optional(optional_kernel & slot) {
        if (!slot.checked) {
            slot.checked = true;
            try {
                if (kernel_files_exist(directory, slot.shape)) {
                    slot.kernel = std::make_unique<xdna_kernel>(device, directory, slot.shape);
                }
            } catch (const std::exception & error) {
                GGML_LOG_WARN("ggml-xdna: loading %s: %s\n", slot.shape.xclbin, error.what());
            }
        }
        return slot.kernel.get();
    }

    xdna_kernel & select_kernel(int64_t m, int64_t k, int64_t n) {
        if (m <= small_shape.m) {
            if (auto * kernel = load_optional(small_kernel)) {
                return *kernel;
            }
        }
        if (k >= 3 * standard_shape.k) {
            if (auto * kernel = load_optional(wide_kernel)) {
                return *kernel;
            }
        }
        if (n >= output_shape.n) {
            if (auto * kernel = load_optional(output_kernel)) {
                return *kernel;
            }
        }
        return standard;
    }

    const fs::path directory;
    xrt::device device;
    xdna_kernel standard;
    optional_kernel small_kernel{ small_shape };
    optional_kernel wide_kernel{ wide_shape };
    optional_kernel output_kernel{ output_shape };
    uint64_t operations = 0;
    uint64_t dispatches = 0;
    uint64_t single_row_ops = 0;
};

std::shared_ptr<xdna_matmul> get_matmul() {
    static std::mutex mutex;
    static std::weak_ptr<xdna_matmul> shared;
    std::lock_guard<std::mutex> lock(mutex);
    auto engine = shared.lock();
    if (!engine) {
        engine = std::make_shared<xdna_matmul>(kernel_directory());
        shared = engine;
    }
    return engine;
}

const char * backend_get_name(ggml_backend_t) {
    return device_name;
}

void backend_free(ggml_backend_t backend) {
    delete static_cast<std::shared_ptr<xdna_matmul> *>(backend->context);
    delete backend;
}

enum ggml_status backend_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    auto & matmul = **static_cast<std::shared_ptr<xdna_matmul> *>(backend->context);
    std::lock_guard<std::mutex> lock(matmul.mutex);
    try {
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            ggml_tensor * node = ggml_graph_node(graph, i);
            if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || is_layout_op(node->op)) {
                continue;
            }
            if (!supports_matmul(node)) {
                GGML_LOG_ERROR("ggml-xdna: unsupported operation %s\n", ggml_op_desc(node));
                return GGML_STATUS_FAILED;
            }
            matmul.compute(node);
        }
    } catch (const std::exception & error) {
        GGML_LOG_ERROR("ggml-xdna: %s\n", error.what());
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

ggml_backend_t device_init(ggml_backend_dev_t device, const char *) {
    try {
        auto matmul = std::make_unique<std::shared_ptr<xdna_matmul>>(get_matmul());
        ggml_backend_i iface = {};
        iface.get_name = backend_get_name;
        iface.free = backend_free;
        iface.graph_compute = backend_compute;
        auto * backend = new ggml_backend{ &backend_guid, iface, device, matmul.get() };
        matmul.release();
        return backend;
    } catch (const std::exception & error) {
        GGML_LOG_ERROR("ggml-xdna: initialization failed: %s\n", error.what());
        return nullptr;
    }
}

void device_get_memory(ggml_backend_dev_t, size_t * free, size_t * total) {
    // The runtime uses shared system memory.
    *free = *total = 0;
}

ggml_backend_device_i device_interface() {
    ggml_backend_device_i iface = {};
    iface.get_name = [](ggml_backend_dev_t) { return device_name; };
    iface.get_description = [](ggml_backend_dev_t) { return device_description; };
    iface.get_type = [](ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_ACCEL; };
    iface.get_memory = device_get_memory;
    iface.get_props = [](ggml_backend_dev_t, ggml_backend_dev_props * props) {
        *props = {};
        props->name = device_name;
        props->description = device_description;
        props->type = GGML_BACKEND_DEVICE_TYPE_ACCEL;
        props->caps.mmap_support = true;
    };
    iface.init_backend = device_init;
    iface.get_buffer_type = [](ggml_backend_dev_t) { return ggml_backend_cpu_buffer_type(); };
    iface.supports_buft = [](ggml_backend_dev_t, ggml_backend_buffer_type_t buft) {
        return ggml_backend_buft_is_host(buft);
    };
    iface.supports_op = [](ggml_backend_dev_t, const ggml_tensor * op) {
        return is_layout_op(op->op) || supports_matmul(op);
    };
    iface.offload_op = [](ggml_backend_dev_t, const ggml_tensor * op) { return supports_matmul(op); };
    return iface;
}

size_t reg_device_count(ggml_backend_reg_t) {
    return kernel_directory().empty() ? 0 : 1;
}

ggml_backend_dev_t reg_device(ggml_backend_reg_t reg, size_t index) {
    if (index >= reg_device_count(reg)) {
        return nullptr;
    }
    static ggml_backend_device device = { device_interface(), reg, nullptr };
    return &device;
}

bool requires_explicit_selection(ggml_backend_dev_t) {
    return true;
}

void * reg_get_proc_address(ggml_backend_reg_t, const char * name) {
    if (std::strcmp(name, "ggml_backend_dev_requires_explicit_selection") == 0) {
        return reinterpret_cast<void *>(requires_explicit_selection);
    }
    return nullptr;
}

} // namespace

ggml_backend_reg_t ggml_backend_xdna_reg(void) {
    static ggml_backend_reg reg = {
        GGML_BACKEND_API_VERSION,
        { [](ggml_backend_reg_t) { return device_name; }, reg_device_count, reg_device, reg_get_proc_address },
        nullptr,
    };
    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_xdna_reg)
