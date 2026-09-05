// cHiDeScaler-Neo DLSS Neural Rendering Backend Pack ABI v1
#pragma once
#include <stdint.h>

#ifdef __cplusplus
#define NEO_DLSSNR_EXTERN extern "C"
#else
#define NEO_DLSSNR_EXTERN extern
#endif

#ifdef _WIN32
#define NEO_DLSSNR_EXPORT NEO_DLSSNR_EXTERN __declspec(dllexport)
#else
#define NEO_DLSSNR_EXPORT NEO_DLSSNR_EXTERN
#endif

#define NEO_DLSSNR_BRIDGE_ABI 1u

// Optional capability bits. ABI-v1 bridges that predate these optional exports
// remain compatible and are treated as reporting zero capabilities.
#define NEO_DLSSNR_CAP_ZERO_GUIDANCE          (1ull << 0)
#define NEO_DLSSNR_CAP_OPTICAL_FLOW_MOTION    (1ull << 1)
#define NEO_DLSSNR_CAP_DEPTH_GUIDANCE         (1ull << 2)
#define NEO_DLSSNR_CAP_D3D12_SHARED_TEXTURE   (1ull << 3)
#define NEO_DLSSNR_CAP_USER_RUNTIME_SELECTION (1ull << 4)
#define NEO_DLSSNR_CAP_EVAL_OPTIONS           (1ull << 5)
#define NEO_DLSSNR_CAP_ADVANCED_OPTIONS       (1ull << 6)

// The first 20 bytes are the original eval-options-v1 contract. Newer bridges may
// append/read the optional tail when struct_size is large enough; ABI stays 1.
typedef struct NeoDlssNrEvalOptions {
    uint32_t struct_size;
    float intensity;
    float local_tone;
    float local_structure;
    float skin_structure;
    uint32_t style;
    uint32_t use_auto_mask;
    uint32_t ui_correction;
    uint32_t reserved;
} NeoDlssNrEvalOptions;

NEO_DLSSNR_EXPORT int32_t neo_dlssnr_set_options(
    void* context, const NeoDlssNrEvalOptions* options);

typedef struct NeoDlssNrCreateDesc {
    uint32_t struct_size;
    uint32_t api_version;
    uint64_t adapter_luid;
    uint32_t width;
    uint32_t height;
    uint32_t preset;
    // ABI v1 uses reserved words only when reserved[6] contains
    // the NRV2 marker. Older bridges safely ignore all reserved values.
    uint32_t reserved[7];
} NeoDlssNrCreateDesc;

typedef struct NeoDlssNrFrameDesc {
    uint32_t struct_size;
    uint32_t width;
    uint32_t height;
    uint32_t reset_history;
    uint64_t frame_index;
    uint32_t reserved[6];
} NeoDlssNrFrameDesc;

// Required ABI-v1 exports. Return 0 on success. Nonzero always means Neo should
// bypass DLSSNR and keep the original frame. No bridge function may terminate
// the host process.
NEO_DLSSNR_EXPORT uint32_t neo_dlssnr_get_api_version(void);
NEO_DLSSNR_EXPORT int32_t neo_dlssnr_create(
    const NeoDlssNrCreateDesc* desc, void** context_out);
NEO_DLSSNR_EXPORT int32_t neo_dlssnr_process_rgba8(
    void* context,
    const NeoDlssNrFrameDesc* desc,
    const uint8_t* input_rgba8,
    uint32_t input_stride,
    uint8_t* output_rgba8,
    uint32_t output_stride);
NEO_DLSSNR_EXPORT int32_t neo_dlssnr_reset_history(void* context);
NEO_DLSSNR_EXPORT void neo_dlssnr_destroy(void* context);
NEO_DLSSNR_EXPORT uint32_t neo_dlssnr_last_error(
    void* context, char* buffer, uint32_t capacity);

// Optional ABI-v1 extensions. Neo discovers these dynamically; do not bump the
// ABI merely to add them. describe_backend should return a human-readable UTF-8
// diagnostic string such as the active runtime build/profile.
NEO_DLSSNR_EXPORT uint64_t neo_dlssnr_get_capabilities(void);
// Optional: called before create, with a verified absolute UTF-16 runtime path.
NEO_DLSSNR_EXPORT int32_t neo_dlssnr_select_runtime(const uint16_t* path);
NEO_DLSSNR_EXPORT uint32_t neo_dlssnr_describe_backend(
    char* buffer, uint32_t capacity);
