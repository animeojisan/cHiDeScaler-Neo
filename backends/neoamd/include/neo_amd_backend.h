#ifndef CHIDESCALER_NEO_AMD_BACKEND_H
#define CHIDESCALER_NEO_AMD_BACKEND_H

#include <stdint.h>

#ifdef _WIN32
/* hipcc parses this header once for host code and again for device code.
 * __declspec(dllexport/dllimport) is meaningful only to the Windows host pass;
 * suppress it for __HIP_DEVICE_COMPILE__ so gfx12 compilation does not emit
 * misleading "dllexport is not supported" warnings. */
#  if defined(__HIP_DEVICE_COMPILE__)
#    define NEOAMD_API
#  elif defined(NEOAMD_BACKEND_BUILD)
#    define NEOAMD_API __declspec(dllexport)
#  else
#    define NEOAMD_API __declspec(dllimport)
#  endif
#else
#  define NEOAMD_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum {
    NEOAMD_BRIDGE_ABI = 1,
    NEOAMD_VENDOR_AMD = 0x1002,
};

enum NeoAmdCapabilityBits {
    NEOAMD_CAP_DYNAMIC_RESOLUTION = 1ull << 0,
    NEOAMD_CAP_FP16_NATIVE        = 1ull << 1,
    NEOAMD_CAP_GFX12_WMMA         = 1ull << 2,
    NEOAMD_CAP_STRUCTURAL_ROUTING = 1ull << 3,
    NEOAMD_CAP_D3D12_SHARED_BUFFER = 1ull << 4,
    /* Optional ABI-1 extension: the host may fill a Backend-Pack-owned
     * D3D12 shared NCHW FP16 input buffer directly from its GPU texture.
     * This is independent of frame geometry; width/height stay runtime values. */
    NEOAMD_CAP_D3D12_SHARED_INPUT  = 1ull << 5,
    /* Optional ABI-1 extension: TemporalFix accepts the current frame through
     * a Backend-Pack-owned shared NCHW3 FP16 input. The Backend Pack keeps the
     * TRUE-7 history on GPU and writes NCHW3 FP16 output to a shared buffer. */
    NEOAMD_CAP_TEMPORAL_SHARED_IO   = 1ull << 6,
    /* v060: cooperative RIFE/DRBA interpolation may return its final RGBA8
     * midpoint through a Backend-Pack-owned D3D12/HIP shared buffer, avoiding
     * the per-phase GPU->CPU->OpenGL upload round trip. */
    NEOAMD_CAP_INTERP_SHARED_RGBA8   = 1ull << 7,
};

enum NeoAmdSessionKind {
    NEOAMD_SESSION_IMAGE         = 0,
    NEOAMD_SESSION_INTERPOLATION = 1,
    NEOAMD_SESSION_TEMPORAL      = 2,
};

typedef struct NeoAmdAdapterInfo {
    uint32_t struct_size;
    uint32_t vendor_id;
    uint32_t device_id;
    uint32_t reserved0;
    uint64_t capabilities;
    char architecture[32];
    uint64_t reserved[6];
} NeoAmdAdapterInfo;

typedef struct NeoAmdSessionInfo {
    uint32_t struct_size;
    uint32_t kind;
    uint32_t temporal_frames;
    uint32_t min_inputs;
    uint32_t max_inputs;
    uint32_t scale_num;
    uint32_t scale_den;
    uint32_t reserved0;
    uint64_t capabilities;
    uint64_t reserved[6];
} NeoAmdSessionInfo;

/*
 * IMPORTANT: width/height intentionally do not exist here.
 * A model session owns graph routing + packed weights only and must be reusable
 * across resolutions. Building a shape-specific engine/session violates ABI 1.
 */
typedef struct NeoAmdCreateDesc {
    uint32_t struct_size;
    const uint16_t *model_path_utf16;
    uint64_t adapter_luid;
    uint64_t flags;
    uint64_t reserved[6];
} NeoAmdCreateDesc;

typedef struct NeoAmdRunDesc {
    uint32_t struct_size;
    uint32_t width;
    uint32_t height;
    uint32_t input_count;
    uint32_t input_pixel_stride;
    uint32_t output_pixel_stride;
    float phase;
    uint32_t flags;
    uint64_t reserved[4];
} NeoAmdRunDesc;

typedef struct NeoAmdOutputDesc {
    uint32_t struct_size;
    uint32_t width;
    uint32_t height;
    uint64_t required_bytes;
    uint64_t reserved[4];
} NeoAmdOutputDesc;

/* Optional ABI-1 extension. The shared resource contains contiguous NCHW FP16
 * planes and is reusable until its resource_key changes (normally only on a
 * grow/resize). The host owns only duplicated Win32 handles returned by
 * neoamd_get_shared_output_handle; the Backend Pack retains the D3D12 resource. */
enum NeoAmdSharedFormat {
    NEOAMD_SHARED_NCHW_FP16 = 1,
    NEOAMD_SHARED_RGBA8 = 2,
};
typedef struct NeoAmdSharedOutputDesc {
    uint32_t struct_size;
    uint32_t width;
    uint32_t height;
    uint32_t format;
    uint64_t resource_key;
    uint64_t byte_len;
    uint64_t allocation_byte_len;
    uint64_t reserved[4];
} NeoAmdSharedOutputDesc;

/* status == 0 means success. Any non-zero status is an error/fallback signal. */
NEOAMD_API uint32_t neoamd_get_api_version(void);
NEOAMD_API int32_t neoamd_query_adapter(uint64_t adapter_luid, NeoAmdAdapterInfo *out_info);
NEOAMD_API int32_t neoamd_create_session(const NeoAmdCreateDesc *desc,
                                          void **out_session,
                                          NeoAmdSessionInfo *out_info);
NEOAMD_API int32_t neoamd_query_output(void *session,
                                       const NeoAmdRunDesc *desc,
                                       NeoAmdOutputDesc *out_desc);
NEOAMD_API int32_t neoamd_run_u8(void *session,
                                 const NeoAmdRunDesc *desc,
                                 const uint8_t *const *inputs,
                                 uint8_t *output,
                                 uint64_t output_capacity);

/* Optional ABI-1 interpolation extension. Multiple canonical phases are
 * evaluated in one Backend-Pack call so phase-independent input upload and
 * encoder work can be shared. `output_stride_bytes` is the byte distance
 * between output images in `outputs`; phase_count is currently 1..4. */
NEOAMD_API int32_t neoamd_run_interp_many_u8(void *session,
                                             const NeoAmdRunDesc *desc,
                                             const uint8_t *const *inputs,
                                             const float *phases,
                                             uint32_t phase_count,
                                             uint8_t *outputs,
                                             uint64_t output_stride_bytes,
                                             uint64_t output_capacity);

/* Optional ABI-1 cooperative interpolation extension. The host first prepares
 * pair-common input/encoder state, then requests exactly one phase at a time.
 * This lets a renderer present each generated midpoint before authorizing the
 * next phase, while keeping the existing batch entry point as a fallback. */
NEOAMD_API int32_t neoamd_prepare_interp_stream_u8(void *session,
                                                   const NeoAmdRunDesc *desc,
                                                   const uint8_t *const *inputs);
NEOAMD_API int32_t neoamd_run_interp_stream_phase_u8(void *session,
                                                     const NeoAmdRunDesc *desc,
                                                     float phase,
                                                     uint8_t *output,
                                                     uint64_t output_capacity);

/* v058 optional RIFE-only cooperative shared-output lane. The common pair
 * preparation is identical to neoamd_prepare_interp_stream_u8, but each phase
 * writes final RGBA8 directly to a shared D3D12 buffer. Hosts must wait until
 * their GL/D3D consumer has retired the previous read before authorizing the
 * next phase to overwrite the same resource_key. */
NEOAMD_API int32_t neoamd_prepare_interp_stream_shared_rgba8(
    void *session, const NeoAmdRunDesc *desc, const uint8_t *const *inputs,
    NeoAmdSharedOutputDesc *out_desc);
NEOAMD_API int32_t neoamd_run_interp_stream_phase_shared_rgba8(
    void *session, const NeoAmdRunDesc *desc, float phase, uint64_t resource_key);

/* Optional GPU-resident output extension. Hosts must discover these exports
 * dynamically and fall back to neoamd_run_u8 when absent. */
NEOAMD_API int32_t neoamd_prepare_shared_fp16(void *session,
                                              const NeoAmdRunDesc *desc,
                                              NeoAmdSharedOutputDesc *out_desc);
NEOAMD_API int32_t neoamd_get_shared_output_handle(void *session,
                                                   uint64_t resource_key,
                                                   uint64_t *out_win32_handle);
NEOAMD_API int32_t neoamd_run_u8_shared_fp16(void *session,
                                             const NeoAmdRunDesc *desc,
                                             const uint8_t *const *inputs,
                                             uint64_t resource_key);


/* Optional full GPU-resident image path. `input_desc` is always the source
 * geometry in NCHW FP16; `output_desc` is the model output geometry. Both
 * resources are owned by the Backend Pack and survive across frames until a
 * grow/resize or explicit release. */
NEOAMD_API int32_t neoamd_prepare_shared_io_fp16(void *session,
                                                 const NeoAmdRunDesc *desc,
                                                 NeoAmdSharedOutputDesc *input_desc,
                                                 NeoAmdSharedOutputDesc *output_desc);
NEOAMD_API int32_t neoamd_get_shared_input_handle(void *session,
                                                  uint64_t resource_key,
                                                  uint64_t *out_win32_handle);
NEOAMD_API int32_t neoamd_run_shared_io_fp16(void *session,
                                             const NeoAmdRunDesc *desc,
                                             uint64_t input_resource_key,
                                             uint64_t output_resource_key);

/* Optional ABI-1 TemporalFix GPU-resident route. The input resource contains
 * only the current visible-size NCHW3 FP16 frame. The Backend Pack owns the
 * seven-frame ring and reproduces the CPU-visible front-padding semantics. */
NEOAMD_API int32_t neoamd_prepare_temporal_shared_io_fp16(
    void *session, const NeoAmdRunDesc *desc,
    NeoAmdSharedOutputDesc *input_desc, NeoAmdSharedOutputDesc *output_desc);
NEOAMD_API int32_t neoamd_run_temporal_shared_io_fp16(
    void *session, const NeoAmdRunDesc *desc,
    uint64_t input_resource_key, uint64_t output_resource_key);
/* Optional v056 zero-conversion chain extension. `source_resource_key` is a
 * Backend-Pack-owned NCHW FP16 output from another NeoAMD session in this
 * process/on this adapter. The downstream session consumes that tensor directly
 * on HIP, avoiding NCHW->RGBA->NCHW between consecutive NeoAMD stages. */
NEOAMD_API int32_t neoamd_run_peer_shared_io_fp16(
    void *session, const NeoAmdRunDesc *desc,
    uint64_t source_resource_key, uint64_t output_resource_key);
/* Optional ABI-1 warm-up aid. Clears only TemporalFix history while retaining
 * weights, tuned launch geometry, HIP Graphs and shared D3D12 resources. */
NEOAMD_API int32_t neoamd_reset_temporal_history(void *session);
/* Optional switch/teardown aid. The host must detach any imported GL/D3D view
 * before calling this. The next prepare call creates a fresh resource/key. */
NEOAMD_API void neoamd_release_shared_output(void *session);
NEOAMD_API void neoamd_destroy_session(void *session);
NEOAMD_API uint32_t neoamd_last_error(void *session, char *utf8, uint32_t capacity);


/* Optional v062 diagnostic extension. Reports the execution lane selected by
 * the most recent image run. Hosts must discover this export dynamically. */
enum NeoAmdRunRouteBits {
    NEOAMD_RUN_ROUTE_GRAPH_FULL       = 1u << 0,
    NEOAMD_RUN_ROUTE_GRAPH_COMPUTE    = 1u << 1,
    NEOAMD_RUN_ROUTE_DIRECT           = 1u << 2,
    NEOAMD_RUN_ROUTE_RTMOSR_FP16      = 1u << 3,
    NEOAMD_RUN_ROUTE_RTMOSR_U8_FUSED = 1u << 4,
};
NEOAMD_API uint32_t neoamd_get_last_run_route(void *session);
/* Optional v063 timing extension. Returns elapsed GPU time, in milliseconds,
 * from the image-run start marker through the output-ready marker. */
NEOAMD_API float neoamd_get_last_image_gpu_ms(void *session);
/* Optional v064 split timing extension. For RTMoSR ordinary U8 ingress these
 * report H2D copy-engine time and compute-graph time separately. Other image
 * families may return 0 until they opt into split timing. */
NEOAMD_API float neoamd_get_last_image_copy_gpu_ms(void *session);
NEOAMD_API float neoamd_get_last_image_compute_gpu_ms(void *session);

/* Optional. Return UTF-8 byte count (NUL may be included or omitted). */
NEOAMD_API uint32_t neoamd_describe_backend(char *utf8, uint32_t capacity);

#ifdef __cplusplus
}
#endif
#endif
