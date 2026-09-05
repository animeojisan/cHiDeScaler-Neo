// SPDX-License-Identifier: MIT
//
// cHiDeScaler-Neo DLSS Neural Rendering Backend Pack - Feature 18 bridge.
// This file is intentionally isolated from the Neo host executable.
// Architecture/compatibility research was informed by the MIT-licensed
// SAOG0721/DaVinci-Resolve-DLSS5 experimental implementation. See
// THIRD_PARTY_NOTICES.md and LICENSE.resolve-dlss5.txt.

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <nvsdk_ngx.h>

#include "../include/neo_dlssnr_backend.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <exception>
#include <iterator>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <vector>

namespace {
using Microsoft::WRL::ComPtr;

constexpr NVSDK_NGX_Feature kFeatureDlssNr = static_cast<NVSDK_NGX_Feature>(18);
// Feature-18 private runtime compatibility contract used by current public
// experimental implementations. It is deliberately confined to this bridge.
constexpr unsigned long long kSnippetApplicationId = 0x0876232Cull;
// Neo uses its own GUID-like NGX project ID for the public/core initialization.
constexpr char kProjectId[] = "9b3c1509-bd46-46d6-91c8-1a100ec619b8";
constexpr char kEngineVersion[] = "cHiDeScaler-Neo-DLSSNR-1";
constexpr wchar_t kRuntimeName[] = L"nvngx_dlssnr.dll";
constexpr uint32_t kCreateOptionsV2Magic = 0x3256524Eu; // ASCII "NRV2" little-endian
constexpr uint32_t kEvalOptionsV1Size = static_cast<uint32_t>(offsetof(NeoDlssNrEvalOptions, style));

using SnippetInitFn = NVSDK_NGX_Result(NVSDK_CONV*)(
    unsigned long long, const wchar_t*, ID3D12Device*, NVSDK_NGX_Version,
    const NVSDK_NGX_Parameter*);
using CreateFeatureFn = NVSDK_NGX_Result(NVSDK_CONV*)(
    ID3D12GraphicsCommandList*, NVSDK_NGX_Feature, NVSDK_NGX_Parameter*,
    NVSDK_NGX_Handle**);
using EvaluateFeatureFn = NVSDK_NGX_Result(NVSDK_CONV*)(
    ID3D12GraphicsCommandList*, const NVSDK_NGX_Handle*,
    const NVSDK_NGX_Parameter*, PFN_NVSDK_NGX_ProgressCallback);
using ReleaseFeatureFn = NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Handle*);
using ShutdownFn = NVSDK_NGX_Result(NVSDK_CONV*)(ID3D12Device*);
using GetModuleFileNameWFn = DWORD(WINAPI*)(HMODULE, LPWSTR, DWORD);

std::mutex g_runtime_mutex;
std::mutex g_hook_mutex;
std::mutex g_error_mutex;
std::string g_last_error;
std::filesystem::path g_selected_runtime;
std::atomic<HMODULE> g_bridge_module{nullptr};
std::atomic<GetModuleFileNameWFn> g_original_get_module_filename{nullptr};
HMODULE g_hooked_runtime = nullptr;
void** g_hooked_iat_slot = nullptr;
uint32_t g_hook_refs = 0;

namespace p {
constexpr char Width[] = "DLSSNR.Width";
constexpr char Height[] = "DLSSNR.Height";
constexpr char InputWidth[] = "DLSSNR.InputWidth";
constexpr char InputHeight[] = "DLSSNR.InputHeight";
constexpr char OutputWidth[] = "DLSSNR.OutputWidth";
constexpr char OutputHeight[] = "DLSSNR.OutputHeight";
constexpr char Upscaling[] = "DLSSNR.Upscaling";
constexpr char Scale[] = "DLSSNR.Scale";
constexpr char ScalingRatio[] = "DLSSNR.ScalingRatio";
constexpr char ScalingRatioCallback[] = "DLSSNRComputeScalingRatioCallback";
constexpr char Preset[] = "DLSSNR.Hint.Render.Preset";
constexpr char Color[] = "DLSSNR.Color";
constexpr char Output[] = "DLSSNR.Output";
constexpr char Motion[] = "DLSSNR.MVec";
constexpr char Depth[] = "DLSSNR.Depth";
constexpr char MotionScaleX[] = "DLSSNR.MVecScaleX";
constexpr char MotionScaleY[] = "DLSSNR.MVecScaleY";
constexpr char DepthInverted[] = "DLSSNR.DepthInverted";
constexpr char Enabled[] = "DLSSNR.Enabled";
constexpr char Reset[] = "DLSSNR.Reset";
constexpr char Style[] = "DLSSNR.Style";
constexpr char Intensity[] = "DLSSNR.Intensity";
constexpr char LocalToneStrength[] = "DLSSNR.LocalToneStrength";
constexpr char LocalStructureStrength[] = "DLSSNR.LocalStructureStrength";
constexpr char SkinStructureStrength[] = "DLSSNR.SkinStructureStrength";
constexpr char UseAutoMask[] = "DLSSNR.UseAutoMask";
constexpr char UiCorrection[] = "DLSSNR.UICorrection";
constexpr char OutputDotWidth[] = "DLSSNR.Output.Width";
constexpr char OutputDotHeight[] = "DLSSNR.Output.Height";
constexpr char IndicatorInvertX[] = "DLSS.Indicator.Invert.X.Axis";
constexpr char IndicatorInvertY[] = "DLSS.Indicator.Invert.Y.Axis";
constexpr char ColorBaseX[] = "DLSSNR.ColorSubrectBaseX";
constexpr char ColorBaseY[] = "DLSSNR.ColorSubrectBaseY";
constexpr char ColorWidth[] = "DLSSNR.ColorSubrectWidth";
constexpr char ColorHeight[] = "DLSSNR.ColorSubrectHeight";
constexpr char OutputBaseX[] = "DLSSNR.OutputSubrectBaseX";
constexpr char OutputBaseY[] = "DLSSNR.OutputSubrectBaseY";
constexpr char OutputRectWidth[] = "DLSSNR.OutputSubrectWidth";
constexpr char OutputRectHeight[] = "DLSSNR.OutputSubrectHeight";
constexpr char MotionBaseX[] = "DLSSNR.MVecSubrectBaseX";
constexpr char MotionBaseY[] = "DLSSNR.MVecSubrectBaseY";
constexpr char MotionWidth[] = "DLSSNR.MVecSubrectWidth";
constexpr char MotionHeight[] = "DLSSNR.MVecSubrectHeight";
constexpr char DepthBaseX[] = "DLSSNR.DepthSubrectBaseX";
constexpr char DepthBaseY[] = "DLSSNR.DepthSubrectBaseY";
constexpr char DepthWidth[] = "DLSSNR.DepthSubrectWidth";
constexpr char DepthHeight[] = "DLSSNR.DepthSubrectHeight";
}

template <typename T>
void* function_address(T fn) noexcept {
    void* result = nullptr;
    static_assert(sizeof(fn) == sizeof(result));
    std::memcpy(&result, &fn, sizeof(result));
    return result;
}

template <typename T>
T get_export(HMODULE module, const char* name) noexcept {
    return reinterpret_cast<T>(GetProcAddress(module, name));
}

bool ngx_ok(NVSDK_NGX_Result r) noexcept { return NVSDK_NGX_SUCCEED(r); }

void set_global_error(const std::string& s) {
    std::scoped_lock lock(g_error_mutex);
    g_last_error = s;
}

std::filesystem::path module_directory() {
    HMODULE module = nullptr;
    const void* address = function_address(&module_directory);
    if (!GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(address), &module)) {
        return {};
    }
    std::wstring path(32768, L'\0');
    const DWORD n = GetModuleFileNameW(module, path.data(), static_cast<DWORD>(path.size()));
    if (!n || n >= path.size()) return {};
    path.resize(n);
    return std::filesystem::path(path).parent_path();
}

std::filesystem::path data_directory() {
    return module_directory() / L"ngx-cache";
}

std::filesystem::path choose_runtime_path() {
    if (!g_selected_runtime.empty()) return g_selected_runtime;
    const auto root = module_directory() / L"runtime";
    const std::filesystem::path candidates[] = {
        root / kRuntimeName,
        root / L"community" / kRuntimeName,
        root / L"mod" / kRuntimeName,
    };
    std::error_code ec;
    for (const auto& path : candidates) {
        if (std::filesystem::is_regular_file(path, ec)) return path;
        ec.clear();
    }
    return root / kRuntimeName;
}

uint64_t pack_luid(const LUID& luid) noexcept {
    return (static_cast<uint64_t>(static_cast<uint32_t>(luid.HighPart)) << 32) |
           static_cast<uint32_t>(luid.LowPart);
}

D3D12_HEAP_PROPERTIES heap_properties(D3D12_HEAP_TYPE type) noexcept {
    D3D12_HEAP_PROPERTIES p{};
    p.Type = type;
    p.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    p.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    p.CreationNodeMask = 1;
    p.VisibleNodeMask = 1;
    return p;
}

D3D12_RESOURCE_DESC texture_desc(uint32_t width, uint32_t height,
                                 DXGI_FORMAT format,
                                 D3D12_RESOURCE_FLAGS flags) noexcept {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Alignment = 0;
    d.Width = width;
    d.Height = height;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.Format = format;
    d.SampleDesc.Count = 1;
    d.SampleDesc.Quality = 0;
    d.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    d.Flags = flags;
    return d;
}

D3D12_RESOURCE_DESC buffer_desc(uint64_t bytes) noexcept {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = bytes;
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.Format = DXGI_FORMAT_UNKNOWN;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return d;
}

D3D12_RESOURCE_BARRIER transition(ID3D12Resource* resource,
                                  D3D12_RESOURCE_STATES before,
                                  D3D12_RESOURCE_STATES after) noexcept {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    b.Transition.pResource = resource;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter = after;
    return b;
}

LONG seh_filter(DWORD code, DWORD* out) noexcept {
    if (out) *out = code;
    return EXCEPTION_EXECUTE_HANDLER;
}

NVSDK_NGX_Result safe_core_init(const wchar_t* data, ID3D12Device* device,
                                const NVSDK_NGX_FeatureCommonInfo* info,
                                DWORD* seh) noexcept {
    *seh = 0;
    __try {
        return NVSDK_NGX_D3D12_Init_with_ProjectID(
            kProjectId, NVSDK_NGX_ENGINE_TYPE_CUSTOM, kEngineVersion,
            data, device, info, NVSDK_NGX_Version_API);
    } __except (seh_filter(GetExceptionCode(), seh)) {
        return NVSDK_NGX_Result_FAIL_PlatformError;
    }
}

NVSDK_NGX_Result safe_alloc_params(NVSDK_NGX_Parameter** out, DWORD* seh) noexcept {
    *seh = 0;
    __try { return NVSDK_NGX_D3D12_AllocateParameters(out); }
    __except (seh_filter(GetExceptionCode(), seh)) {
        return NVSDK_NGX_Result_FAIL_PlatformError;
    }
}

NVSDK_NGX_Result safe_destroy_params(NVSDK_NGX_Parameter* p, DWORD* seh) noexcept {
    *seh = 0;
    __try { return NVSDK_NGX_D3D12_DestroyParameters(p); }
    __except (seh_filter(GetExceptionCode(), seh)) {
        return NVSDK_NGX_Result_FAIL_PlatformError;
    }
}

NVSDK_NGX_Result safe_snippet_init(SnippetInitFn fn, const wchar_t* data,
                                   ID3D12Device* device, DWORD* seh) noexcept {
    *seh = 0;
    __try { return fn(kSnippetApplicationId, data, device, NVSDK_NGX_Version_API, nullptr); }
    __except (seh_filter(GetExceptionCode(), seh)) {
        return NVSDK_NGX_Result_FAIL_PlatformError;
    }
}

NVSDK_NGX_Result safe_create(CreateFeatureFn fn, ID3D12GraphicsCommandList* list,
                             NVSDK_NGX_Parameter* params, NVSDK_NGX_Handle** handle,
                             DWORD* seh) noexcept {
    *seh = 0;
    __try { return fn(list, kFeatureDlssNr, params, handle); }
    __except (seh_filter(GetExceptionCode(), seh)) {
        return NVSDK_NGX_Result_FAIL_PlatformError;
    }
}

NVSDK_NGX_Result safe_evaluate(EvaluateFeatureFn fn,
                               ID3D12GraphicsCommandList* list,
                               const NVSDK_NGX_Handle* handle,
                               const NVSDK_NGX_Parameter* params,
                               DWORD* seh) noexcept {
    *seh = 0;
    __try { return fn(list, handle, params, nullptr); }
    __except (seh_filter(GetExceptionCode(), seh)) {
        return NVSDK_NGX_Result_FAIL_PlatformError;
    }
}

NVSDK_NGX_Result safe_release(ReleaseFeatureFn fn, NVSDK_NGX_Handle* handle,
                              DWORD* seh) noexcept {
    *seh = 0;
    __try { return fn(handle); }
    __except (seh_filter(GetExceptionCode(), seh)) {
        return NVSDK_NGX_Result_FAIL_PlatformError;
    }
}

NVSDK_NGX_Result safe_shutdown(ShutdownFn fn, ID3D12Device* device,
                               DWORD* seh) noexcept {
    *seh = 0;
    __try { return fn(device); }
    __except (seh_filter(GetExceptionCode(), seh)) {
        return NVSDK_NGX_Result_FAIL_PlatformError;
    }
}

NVSDK_NGX_Result NVSDK_CONV scaling_ratio_callback(NVSDK_NGX_Parameter* params) noexcept {
    __try {
        if (!params) return NVSDK_NGX_Result_FAIL_InvalidParameter;
        params->Set(p::ScalingRatio, 1.0F);
        return NVSDK_NGX_Result_Success;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return NVSDK_NGX_Result_FAIL_PlatformError;
    }
}

bool safe_set_create_params(NVSDK_NGX_Parameter* params, uint32_t width,
                            uint32_t height, int preset,
                            const NeoDlssNrEvalOptions* options,
                            DWORD* seh) noexcept {
    *seh = 0;
    __try {
        params->Set(p::Width, width);
        params->Set(p::Height, height);
        params->Set(p::InputWidth, width);
        params->Set(p::InputHeight, height);
        params->Set(p::OutputWidth, width);
        params->Set(p::OutputHeight, height);
        params->Set(p::OutputDotWidth, width);
        params->Set(p::OutputDotHeight, height);
        params->Set(p::Upscaling, 0U);
        params->Set(p::Scale, 1.0F);
        params->Set(p::ScalingRatio, 1.0F);
        params->Set(p::ScalingRatioCallback, function_address(&scaling_ratio_callback));
        params->Set(p::Preset, preset);
        params->Set(NVSDK_NGX_Parameter_Width, width);
        params->Set(NVSDK_NGX_Parameter_Height, height);
        params->Set(NVSDK_NGX_Parameter_PerfQualityValue,
                    static_cast<int>(NVSDK_NGX_PerfQuality_Value_Balanced));
        params->Set(NVSDK_NGX_Parameter_CreationNodeMask, 1U);
        params->Set(NVSDK_NGX_Parameter_VisibilityNodeMask, 1U);
        if (options) {
            params->Set(p::Style, static_cast<int>(options->style));
            params->Set(p::Intensity, options->intensity);
            params->Set(p::LocalToneStrength, options->local_tone);
            params->Set(p::LocalStructureStrength, options->local_structure);
            if (options->skin_structure >= 0.0F)
                params->Set(p::SkinStructureStrength, options->skin_structure);
            params->Set(p::UseAutoMask, options->use_auto_mask ? 1 : 0);
            params->Set(p::UiCorrection, options->ui_correction ? 1 : 0);
        }
        return true;
    } __except (seh_filter(GetExceptionCode(), seh)) {
        return false;
    }
}

void set_rect(NVSDK_NGX_Parameter* params, const char* bx, const char* by,
              const char* w, const char* h, uint32_t width, uint32_t height) {
    params->Set(bx, 0U); params->Set(by, 0U);
    params->Set(w, width); params->Set(h, height);
}

bool safe_set_eval_params(NVSDK_NGX_Parameter* params,
                          ID3D12Resource* color, ID3D12Resource* output,
                          ID3D12Resource* motion, ID3D12Resource* depth,
                          uint32_t width, uint32_t height, bool reset,
                          const NeoDlssNrEvalOptions* options,
                          DWORD* seh) noexcept {
    *seh = 0;
    __try {
        params->Set(p::Color, color);
        params->Set(p::Output, output);
        params->Set(p::Motion, motion);
        params->Set(p::Depth, depth);
        set_rect(params, p::ColorBaseX, p::ColorBaseY, p::ColorWidth, p::ColorHeight, width, height);
        set_rect(params, p::OutputBaseX, p::OutputBaseY, p::OutputRectWidth, p::OutputRectHeight, width, height);
        set_rect(params, p::MotionBaseX, p::MotionBaseY, p::MotionWidth, p::MotionHeight, width, height);
        set_rect(params, p::DepthBaseX, p::DepthBaseY, p::DepthWidth, p::DepthHeight, width, height);
        params->Set(p::MotionScaleX, 1.0F);
        params->Set(p::MotionScaleY, 1.0F);
        params->Set(p::DepthInverted, 1);
        params->Set(p::IndicatorInvertX, 0);
        params->Set(p::IndicatorInvertY, 0);
        params->Set(p::Enabled, 1);
        params->Set(p::Reset, reset ? 1 : 0);
        params->Set(p::Style, static_cast<int>(options->style));
        params->Set(p::Intensity, options->intensity);
        params->Set(p::LocalToneStrength, options->local_tone);
        params->Set(p::LocalStructureStrength, options->local_structure);
        if (options->skin_structure >= 0.0F)
            params->Set(p::SkinStructureStrength, options->skin_structure);
        params->Set(p::UseAutoMask, options->use_auto_mask ? 1 : 0);
        params->Set(p::UiCorrection, options->ui_correction ? 1 : 0);
        return true;
    } __except (seh_filter(GetExceptionCode(), seh)) {
        return false;
    }
}

DWORD WINAPI hooked_get_module_filename(HMODULE module, LPWSTR filename,
                                        DWORD size) noexcept {
    if (module == g_bridge_module.load(std::memory_order_acquire)) {
        constexpr wchar_t authorized[] = L"nvngx.dll";
        constexpr DWORD length = static_cast<DWORD>(std::size(authorized) - 1);
        if (!filename || size == 0) {
            SetLastError(ERROR_INSUFFICIENT_BUFFER);
            return 0;
        }
        if (size <= length) {
            const DWORD copy = size > 0 ? size - 1 : 0;
            if (copy) std::memcpy(filename, authorized, copy * sizeof(wchar_t));
            filename[size - 1] = L'\0';
            SetLastError(ERROR_INSUFFICIENT_BUFFER);
            return size;
        }
        std::memcpy(filename, authorized, sizeof(authorized));
        return length;
    }
    const auto original = g_original_get_module_filename.load(std::memory_order_acquire);
    if (original) return original(module, filename, size);
    SetLastError(ERROR_INVALID_FUNCTION);
    return 0;
}

void** find_iat_slot(HMODULE module, const char* function_name) noexcept {
    if (!module || !function_name) return nullptr;
    auto* base = reinterpret_cast<std::byte*>(module);
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
    if (dos->e_magic != IMAGE_DOS_SIGNATURE || dos->e_lfanew <= 0) return nullptr;
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE ||
        nt->OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR64_MAGIC) return nullptr;
    const auto dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (!dir.VirtualAddress || !dir.Size || dir.VirtualAddress >= nt->OptionalHeader.SizeOfImage)
        return nullptr;
    auto* desc = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(base + dir.VirtualAddress);
    const auto* end = reinterpret_cast<const IMAGE_IMPORT_DESCRIPTOR*>(base + dir.VirtualAddress + dir.Size);
    for (; desc < end && desc->Name; ++desc) {
        if (desc->Name >= nt->OptionalHeader.SizeOfImage) continue;
        const char* lib = reinterpret_cast<const char*>(base + desc->Name);
        if (_stricmp(lib, "KERNEL32.dll") != 0 &&
            _stricmp(lib, "api-ms-win-core-libraryloader-l1-2-0.dll") != 0 &&
            _stricmp(lib, "api-ms-win-core-libraryloader-l1-1-0.dll") != 0) continue;
        if (!desc->OriginalFirstThunk || !desc->FirstThunk) continue;
        auto* names = reinterpret_cast<IMAGE_THUNK_DATA64*>(base + desc->OriginalFirstThunk);
        auto* addrs = reinterpret_cast<IMAGE_THUNK_DATA64*>(base + desc->FirstThunk);
        for (; names->u1.AddressOfData; ++names, ++addrs) {
            if (IMAGE_SNAP_BY_ORDINAL64(names->u1.Ordinal)) continue;
            const auto rva = static_cast<uint32_t>(names->u1.AddressOfData);
            if (rva >= nt->OptionalHeader.SizeOfImage) return nullptr;
            const auto* imp = reinterpret_cast<const IMAGE_IMPORT_BY_NAME*>(base + rva);
            if (std::strcmp(reinterpret_cast<const char*>(imp->Name), function_name) == 0)
                return reinterpret_cast<void**>(&addrs->u1.Function);
        }
    }
    return nullptr;
}

bool install_caller_hook(HMODULE runtime, void*** slot_out) noexcept {
    std::scoped_lock lock(g_hook_mutex);
    void** slot = find_iat_slot(runtime, "GetModuleFileNameW");
    if (!slot) return false;
    if (g_hook_refs) {
        if (runtime != g_hooked_runtime || slot != g_hooked_iat_slot) return false;
        ++g_hook_refs;
        *slot_out = slot;
        return true;
    }
    HMODULE bridge = nullptr;
    const void* hook_addr = function_address(&hooked_get_module_filename);
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(hook_addr), &bridge)) return false;
    DWORD old = 0;
    if (!VirtualProtect(slot, sizeof(void*), PAGE_READWRITE, &old)) return false;
    g_bridge_module.store(bridge, std::memory_order_release);
    void* previous = InterlockedExchangePointer(
        reinterpret_cast<void* volatile*>(slot), const_cast<void*>(hook_addr));
    GetModuleFileNameWFn original = nullptr;
    std::memcpy(&original, &previous, sizeof(original));
    g_original_get_module_filename.store(original, std::memory_order_release);
    DWORD ignored = 0;
    VirtualProtect(slot, sizeof(void*), old, &ignored);
    FlushInstructionCache(GetCurrentProcess(), slot, sizeof(void*));
    if (!original) return false;
    g_hooked_runtime = runtime;
    g_hooked_iat_slot = slot;
    g_hook_refs = 1;
    *slot_out = slot;
    return true;
}

bool restore_caller_hook(void** slot) noexcept {
    if (!slot) return true;
    std::scoped_lock lock(g_hook_mutex);
    if (!g_hook_refs || slot != g_hooked_iat_slot) return false;
    if (g_hook_refs > 1) {
        --g_hook_refs;
        return true;
    }
    const auto original = g_original_get_module_filename.load(std::memory_order_acquire);
    DWORD old = 0;
    if (!original || !VirtualProtect(slot, sizeof(void*), PAGE_READWRITE, &old)) return false;
    InterlockedExchangePointer(reinterpret_cast<void* volatile*>(slot), function_address(original));
    DWORD ignored = 0;
    VirtualProtect(slot, sizeof(void*), old, &ignored);
    FlushInstructionCache(GetCurrentProcess(), slot, sizeof(void*));
    g_original_get_module_filename.store(nullptr, std::memory_order_release);
    g_bridge_module.store(nullptr, std::memory_order_release);
    g_hooked_runtime = nullptr;
    g_hooked_iat_slot = nullptr;
    g_hook_refs = 0;
    return true;
}

struct Context {
    NeoDlssNrEvalOptions options{sizeof(NeoDlssNrEvalOptions), 1.0F, 1.0F, 1.0F, 1.0F, 0U, 0U, 0U, 0U};
    std::mutex mutex;
    std::string last_error;
    std::filesystem::path runtime_path;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t preset = 1;
    uint64_t requested_luid = 0;
    bool reset_pending = true;
    bool initialized = false;
    bool core_initialized = false;
    bool snippet_initialized = false;

    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    HANDLE fence_event = nullptr;
    uint64_t fence_value = 0;

    ComPtr<ID3D12Resource> input;
    ComPtr<ID3D12Resource> output;
    ComPtr<ID3D12Resource> motion;
    ComPtr<ID3D12Resource> depth;
    ComPtr<ID3D12Resource> upload;
    ComPtr<ID3D12Resource> readback;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT input_fp{};
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT output_fp{};
    uint8_t* mapped_upload = nullptr;

    NVSDK_NGX_Parameter* params = nullptr;
    NVSDK_NGX_Handle* feature = nullptr;
    HMODULE runtime = nullptr;
    SnippetInitFn snippet_init = nullptr;
    CreateFeatureFn snippet_create = nullptr;
    EvaluateFeatureFn snippet_evaluate = nullptr;
    ReleaseFeatureFn snippet_release = nullptr;
    ShutdownFn snippet_shutdown = nullptr;
    void** hook_slot = nullptr;

    bool fail(const std::string& msg) {
        last_error = msg;
        set_global_error(msg);
        return false;
    }
    bool fail_hr(const char* op, HRESULT hr) {
        char buf[256]{};
        std::snprintf(buf, sizeof(buf), "%s failed (HRESULT 0x%08X)", op,
                      static_cast<unsigned>(hr));
        return fail(buf);
    }
    bool fail_ngx(const char* op, NVSDK_NGX_Result r, DWORD seh) {
        char buf[320]{};
        if (seh) std::snprintf(buf, sizeof(buf), "%s raised SEH 0x%08X", op,
                               static_cast<unsigned>(seh));
        else std::snprintf(buf, sizeof(buf), "%s failed (NGX 0x%08X)", op,
                           static_cast<unsigned>(r));
        return fail(buf);
    }

    bool choose_device() {
        ComPtr<IDXGIFactory6> factory;
        HRESULT hr = CreateDXGIFactory2(0, IID_PPV_ARGS(factory.ReleaseAndGetAddressOf()));
        if (FAILED(hr)) return fail_hr("CreateDXGIFactory2", hr);
        for (UINT i = 0;; ++i) {
            ComPtr<IDXGIAdapter1> adapter;
            hr = factory->EnumAdapterByGpuPreference(
                i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                IID_PPV_ARGS(adapter.ReleaseAndGetAddressOf()));
            if (hr == DXGI_ERROR_NOT_FOUND) break;
            if (FAILED(hr)) continue;
            DXGI_ADAPTER_DESC1 desc{};
            if (FAILED(adapter->GetDesc1(&desc)) || desc.VendorId != 0x10DE ||
                (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) continue;
            if (requested_luid && pack_luid(desc.AdapterLuid) != requested_luid) continue;
            hr = D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0,
                                   IID_PPV_ARGS(device.ReleaseAndGetAddressOf()));
            if (SUCCEEDED(hr)) return true;
        }
        if (requested_luid) return fail("Requested NVIDIA DXGI adapter LUID is unavailable for D3D12");
        return fail("No compatible NVIDIA D3D12 adapter was found");
    }

    bool create_d3d() {
        if (!choose_device()) return false;
        D3D12_COMMAND_QUEUE_DESC q{};
        q.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        HRESULT hr = device->CreateCommandQueue(&q, IID_PPV_ARGS(queue.ReleaseAndGetAddressOf()));
        if (SUCCEEDED(hr)) hr = device->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(allocator.ReleaseAndGetAddressOf()));
        if (SUCCEEDED(hr)) hr = device->CreateCommandList(
            0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
            IID_PPV_ARGS(list.ReleaseAndGetAddressOf()));
        if (SUCCEEDED(hr)) hr = device->CreateFence(
            0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(fence.ReleaseAndGetAddressOf()));
        if (FAILED(hr)) return fail_hr("Create D3D12 command objects", hr);
        fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        return fence_event != nullptr || fail("CreateEventW for D3D12 fence failed");
    }

    bool create_texture(DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags,
                        D3D12_RESOURCE_STATES state, ComPtr<ID3D12Resource>& out) {
        const auto heap = heap_properties(D3D12_HEAP_TYPE_DEFAULT);
        const auto desc = texture_desc(width, height, format, flags);
        const HRESULT hr = device->CreateCommittedResource(
            &heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
            IID_PPV_ARGS(out.ReleaseAndGetAddressOf()));
        return SUCCEEDED(hr) || fail_hr("Create D3D12 texture", hr);
    }

    bool create_buffer(uint64_t bytes, D3D12_HEAP_TYPE type,
                       D3D12_RESOURCE_STATES state, ComPtr<ID3D12Resource>& out) {
        const auto heap = heap_properties(type);
        const auto desc = buffer_desc(bytes);
        const HRESULT hr = device->CreateCommittedResource(
            &heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
            IID_PPV_ARGS(out.ReleaseAndGetAddressOf()));
        return SUCCEEDED(hr) || fail_hr("Create D3D12 buffer", hr);
    }

    bool create_frame_resources() {
        if (!create_texture(DXGI_FORMAT_R8G8B8A8_UNORM,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_DEST, input) ||
            !create_texture(DXGI_FORMAT_R8G8B8A8_UNORM,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON, output) ||
            !create_texture(DXGI_FORMAT_R16G16_FLOAT, D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COPY_DEST, motion) ||
            !create_texture(DXGI_FORMAT_R32_FLOAT, D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COPY_DEST, depth)) return false;

        UINT rows = 0; UINT64 row_bytes = 0; UINT64 total = 0;
        auto d = input->GetDesc();
        device->GetCopyableFootprints(&d, 0, 1, 0, &input_fp, &rows, &row_bytes, &total);
        if (!create_buffer(total, D3D12_HEAP_TYPE_UPLOAD,
                           D3D12_RESOURCE_STATE_GENERIC_READ, upload)) return false;
        D3D12_RANGE no_read{0, 0};
        void* mapped = nullptr;
        HRESULT hr = upload->Map(0, &no_read, &mapped);
        if (FAILED(hr) || !mapped) return fail_hr("Map input upload", hr);
        mapped_upload = static_cast<uint8_t*>(mapped);

        rows = 0; row_bytes = 0; total = 0;
        d = output->GetDesc();
        device->GetCopyableFootprints(&d, 0, 1, 0, &output_fp, &rows, &row_bytes, &total);
        return create_buffer(total, D3D12_HEAP_TYPE_READBACK,
                             D3D12_RESOURCE_STATE_COPY_DEST, readback);
    }

    bool wait_queue() {
        const uint64_t value = ++fence_value;
        HRESULT hr = queue->Signal(fence.Get(), value);
        if (FAILED(hr)) return fail_hr("Signal D3D12 fence", hr);
        if (fence->GetCompletedValue() >= value) return true;
        ResetEvent(fence_event);
        hr = fence->SetEventOnCompletion(value, fence_event);
        if (FAILED(hr)) return fail_hr("SetEventOnCompletion", hr);
        return WaitForSingleObject(fence_event, 5000) == WAIT_OBJECT_0 ||
               fail("Wait for D3D12 fence failed");
    }

    bool execute_wait() {
        HRESULT hr = list->Close();
        if (FAILED(hr)) return fail_hr("Close D3D12 command list", hr);
        ID3D12CommandList* lists[]{list.Get()};
        queue->ExecuteCommandLists(1, lists);
        return wait_queue();
    }

    bool reset_list() {
        HRESULT hr = allocator->Reset();
        if (SUCCEEDED(hr)) hr = list->Reset(allocator.Get(), nullptr);
        return SUCCEEDED(hr) || fail_hr("Reset D3D12 command list", hr);
    }

    bool zero_texture(ID3D12Resource* tex, ComPtr<ID3D12Resource>& temp_upload) {
        const auto d = tex->GetDesc();
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
        UINT rows = 0; UINT64 row_bytes = 0; UINT64 total = 0;
        device->GetCopyableFootprints(&d, 0, 1, 0, &fp, &rows, &row_bytes, &total);
        if (!create_buffer(total, D3D12_HEAP_TYPE_UPLOAD,
                           D3D12_RESOURCE_STATE_GENERIC_READ, temp_upload)) return false;
        void* mapped = nullptr; D3D12_RANGE no_read{0, 0};
        HRESULT hr = temp_upload->Map(0, &no_read, &mapped);
        if (FAILED(hr) || !mapped) return fail_hr("Map zero guidance upload", hr);
        std::memset(mapped, 0, static_cast<size_t>(total));
        D3D12_RANGE written{0, static_cast<SIZE_T>(total)};
        temp_upload->Unmap(0, &written);
        D3D12_TEXTURE_COPY_LOCATION dst{}; dst.pResource = tex;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        D3D12_TEXTURE_COPY_LOCATION src{}; src.pResource = temp_upload.Get();
        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; src.PlacedFootprint = fp;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        auto b = transition(tex, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
        list->ResourceBarrier(1, &b);
        return true;
    }

    bool initialize_ngx() {
        runtime_path = choose_runtime_path();
        if (!std::filesystem::is_regular_file(runtime_path))
            return fail("runtime/nvngx_dlssnr.dll was not found");
        const auto data = data_directory();
        std::error_code ec; std::filesystem::create_directories(data, ec);
        const auto runtime_dir = runtime_path.parent_path();
        const std::wstring feature_path = runtime_dir.wstring();
        const wchar_t* paths[]{feature_path.c_str()};
        NVSDK_NGX_FeatureCommonInfo info{};
        info.PathListInfo.Path = paths;
        info.PathListInfo.Length = 1;
        DWORD seh = 0;
        auto result = safe_core_init(data.c_str(), device.Get(), &info, &seh);
        if (seh || !ngx_ok(result)) return fail_ngx("NGX Core Init", result, seh);
        core_initialized = true;

        runtime = LoadLibraryExW(runtime_path.c_str(), nullptr,
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (!runtime) return fail("LoadLibraryExW for private nvngx_dlssnr.dll failed");
        snippet_init = get_export<SnippetInitFn>(runtime, "NVSDK_NGX_D3D12_Init_Ext");
        snippet_create = get_export<CreateFeatureFn>(runtime, "NVSDK_NGX_D3D12_CreateFeature");
        snippet_evaluate = get_export<EvaluateFeatureFn>(runtime, "NVSDK_NGX_D3D12_EvaluateFeature");
        snippet_release = get_export<ReleaseFeatureFn>(runtime, "NVSDK_NGX_D3D12_ReleaseFeature");
        snippet_shutdown = get_export<ShutdownFn>(runtime, "NVSDK_NGX_D3D12_Shutdown1");
        if (!snippet_init || !snippet_create || !snippet_evaluate || !snippet_release || !snippet_shutdown)
            return fail("Private DLSSNR runtime exports are incomplete");
        if (!install_caller_hook(runtime, &hook_slot))
            return fail("DLSSNR caller compatibility hook could not be installed");
        result = safe_snippet_init(snippet_init, data.c_str(), device.Get(), &seh);
        if (seh || !ngx_ok(result)) return fail_ngx("DLSSNR Init_Ext", result, seh);
        snippet_initialized = true;
        result = safe_alloc_params(&params, &seh);
        if (seh || !ngx_ok(result) || !params) return fail_ngx("NGX AllocateParameters", result, seh);
        if (!safe_set_create_params(params, width, height, static_cast<int>(preset), &options, &seh))
            return fail_ngx("Set Feature 18 create parameters", NVSDK_NGX_Result_FAIL_PlatformError, seh);

        ComPtr<ID3D12Resource> zero_motion, zero_depth;
        if (!zero_texture(motion.Get(), zero_motion) || !zero_texture(depth.Get(), zero_depth)) return false;
        result = safe_create(snippet_create, list.Get(), params, &feature, &seh);
        if (seh || !ngx_ok(result) || !feature) return fail_ngx("Feature 18 CreateFeature", result, seh);
        if (!execute_wait()) return false;
        return true;
    }

    bool initialize() {
        if (width == 0 || height == 0) return fail("Invalid zero-sized Feature 18 session");
        if (!create_d3d() || !create_frame_resources() || !initialize_ngx()) {
            const std::string e = last_error;
            shutdown();
            last_error = e;
            set_global_error(e);
            return false;
        }
        initialized = true;
        reset_pending = true;
        last_error.clear();
        return true;
    }

    bool upload_rgba8(const uint8_t* src, uint32_t stride) {
        const uint32_t row = width * 4u;
        if (!src || stride < row) return fail("Invalid RGBA8 input stride");
        for (uint32_t y = 0; y < height; ++y) {
            std::memcpy(mapped_upload + static_cast<size_t>(y) * input_fp.Footprint.RowPitch,
                        src + static_cast<size_t>(y) * stride, row);
        }
        return true;
    }

    bool download_rgba8(const uint8_t* original, uint32_t input_stride,
                        uint8_t* dst, uint32_t output_stride) {
        const uint32_t row = width * 4u;
        if (!dst || output_stride < row) return fail("Invalid RGBA8 output stride");
        const uint64_t bytes = static_cast<uint64_t>(output_fp.Footprint.RowPitch) * height;
        D3D12_RANGE read_range{0, static_cast<SIZE_T>(bytes)};
        void* mapped = nullptr;
        HRESULT hr = readback->Map(0, &read_range, &mapped);
        if (FAILED(hr) || !mapped) return fail_hr("Map Feature 18 output", hr);
        const auto* base = static_cast<const uint8_t*>(mapped);
        for (uint32_t y = 0; y < height; ++y) {
            const auto* src_row = base + static_cast<size_t>(y) * output_fp.Footprint.RowPitch;
            auto* dst_row = dst + static_cast<size_t>(y) * output_stride;
            std::memcpy(dst_row, src_row, row);
            // Preserve host alpha. Feature 18 is used only as RGB enhancement.
            if (original) {
                const auto* in_row = original + static_cast<size_t>(y) * input_stride;
                for (uint32_t x = 0; x < width; ++x) dst_row[x * 4u + 3u] = in_row[x * 4u + 3u];
            }
        }
        D3D12_RANGE no_write{0, 0};
        readback->Unmap(0, &no_write);
        return true;
    }

    bool process(const NeoDlssNrFrameDesc* frame, const uint8_t* src,
                 uint32_t input_stride, uint8_t* dst, uint32_t output_stride) {
        if (!initialized || !feature || !params) return fail("DLSSNR session is not initialized");
        if (!frame || frame->struct_size < sizeof(NeoDlssNrFrameDesc) ||
            frame->width != width || frame->height != height)
            return fail("Frame geometry does not match DLSSNR session");
        if (!upload_rgba8(src, input_stride) || !reset_list()) return false;

        D3D12_TEXTURE_COPY_LOCATION input_dst{}; input_dst.pResource = input.Get();
        input_dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        D3D12_TEXTURE_COPY_LOCATION input_src{}; input_src.pResource = upload.Get();
        input_src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; input_src.PlacedFootprint = input_fp;
        list->CopyTextureRegion(&input_dst, 0, 0, 0, &input_src, nullptr);

        D3D12_RESOURCE_BARRIER before[]{
            transition(input.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            transition(output.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            transition(motion.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            transition(depth.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
        };
        list->ResourceBarrier(static_cast<UINT>(std::size(before)), before);

        DWORD seh = 0;
        const bool reset = reset_pending || frame->reset_history != 0;
        if (!safe_set_eval_params(params, input.Get(), output.Get(), motion.Get(), depth.Get(),
                                  width, height, reset, &options, &seh)) {
            list->Close();
            return fail_ngx("Set Feature 18 evaluate parameters", NVSDK_NGX_Result_FAIL_PlatformError, seh);
        }
        const auto result = safe_evaluate(snippet_evaluate, list.Get(), feature, params, &seh);
        if (seh || !ngx_ok(result)) {
            list->Close();
            return fail_ngx("Feature 18 EvaluateFeature", result, seh);
        }

        auto out_to_copy = transition(output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                      D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &out_to_copy);
        D3D12_TEXTURE_COPY_LOCATION out_dst{}; out_dst.pResource = readback.Get();
        out_dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; out_dst.PlacedFootprint = output_fp;
        D3D12_TEXTURE_COPY_LOCATION out_src{}; out_src.pResource = output.Get();
        out_src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        list->CopyTextureRegion(&out_dst, 0, 0, 0, &out_src, nullptr);

        D3D12_RESOURCE_BARRIER after[]{
            transition(input.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST),
            transition(output.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON),
            transition(motion.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON),
            transition(depth.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON),
        };
        list->ResourceBarrier(static_cast<UINT>(std::size(after)), after);
        if (!execute_wait() || !download_rgba8(src, input_stride, dst, output_stride)) return false;
        reset_pending = false;
        last_error.clear();
        return true;
    }

    void shutdown() noexcept {
        if (queue && fence) wait_queue();
        if (feature && snippet_release) {
            DWORD seh = 0; safe_release(snippet_release, feature, &seh); feature = nullptr;
        }
        if (params) {
            DWORD seh = 0; safe_destroy_params(params, &seh); params = nullptr;
        }
        if (snippet_initialized && snippet_shutdown && device) {
            DWORD seh = 0; safe_shutdown(snippet_shutdown, device.Get(), &seh);
        }
        snippet_initialized = false;
        const bool restored = restore_caller_hook(hook_slot);
        hook_slot = nullptr;
        if (runtime && restored) FreeLibrary(runtime);
        // If restoration fails, intentionally keep the module loaded so an IAT
        // slot can never point into an unloaded bridge/runtime path.
        runtime = nullptr;
        if (core_initialized && device) {
            DWORD seh = 0;
            safe_shutdown(static_cast<ShutdownFn>(&NVSDK_NGX_D3D12_Shutdown1), device.Get(), &seh);
        }
        core_initialized = false;
        if (mapped_upload && upload) { upload->Unmap(0, nullptr); mapped_upload = nullptr; }
        if (fence_event) { CloseHandle(fence_event); fence_event = nullptr; }
        readback.Reset(); upload.Reset(); depth.Reset(); motion.Reset(); output.Reset(); input.Reset();
        fence.Reset(); list.Reset(); allocator.Reset(); queue.Reset(); device.Reset();
        initialized = false;
    }

    ~Context() { shutdown(); }
};

uint32_t copy_string(const std::string& s, char* buffer, uint32_t capacity) {
    if (!buffer || capacity == 0) return 0;
    const uint32_t n = static_cast<uint32_t>(std::min<size_t>(s.size(), capacity - 1));
    if (n) std::memcpy(buffer, s.data(), n);
    buffer[n] = '\0';
    return n;
}

} // namespace

NEO_DLSSNR_EXPORT uint32_t neo_dlssnr_get_api_version(void) {
    return NEO_DLSSNR_BRIDGE_ABI;
}

NEO_DLSSNR_EXPORT int32_t neo_dlssnr_select_runtime(const uint16_t* path) {
    try {
        if (!path) return 1;
        std::scoped_lock execution(g_runtime_mutex);
        auto root = std::filesystem::canonical(module_directory() / L"runtime");
        auto candidate = std::filesystem::canonical(reinterpret_cast<const wchar_t*>(path));
        auto r = root.begin();
        auto c = candidate.begin();
        for (; r != root.end(); ++r, ++c) {
            if (c == candidate.end() || _wcsicmp(r->c_str(), c->c_str()) != 0) return 2;
        }
        if (!std::filesystem::is_regular_file(candidate)) return 3;
        g_selected_runtime = candidate;
        return 0;
    } catch (...) { return 4; }
}

NEO_DLSSNR_EXPORT uint64_t neo_dlssnr_get_capabilities(void) {
    return NEO_DLSSNR_CAP_ZERO_GUIDANCE | NEO_DLSSNR_CAP_USER_RUNTIME_SELECTION |
           NEO_DLSSNR_CAP_EVAL_OPTIONS | NEO_DLSSNR_CAP_ADVANCED_OPTIONS;
}

NEO_DLSSNR_EXPORT uint32_t neo_dlssnr_describe_backend(char* buffer, uint32_t capacity) {
    try {
        std::string text = "Neo Feature 18 D3D12 backend; zero-guidance; runtime=";
        const auto path = choose_runtime_path();
        text += path.empty() ? std::string("unresolved") : path.filename().string();
        return copy_string(text, buffer, capacity);
    } catch (...) {
        return copy_string(
            "Neo Feature 18 D3D12 backend; zero-guidance; runtime=unresolved",
            buffer,
            capacity);
    }
}

NEO_DLSSNR_EXPORT int32_t neo_dlssnr_create(const NeoDlssNrCreateDesc* desc,
                                             void** context_out) {
    if (context_out) *context_out = nullptr;
    if (!desc || !context_out || desc->struct_size < sizeof(NeoDlssNrCreateDesc) ||
        desc->api_version != NEO_DLSSNR_BRIDGE_ABI || desc->width == 0 || desc->height == 0) {
        set_global_error("Invalid NeoDlssNrCreateDesc");
        return 10;
    }
    std::unique_ptr<Context> ctx(new (std::nothrow) Context{});
    if (!ctx) { set_global_error("Out of memory creating DLSSNR context"); return 11; }
    ctx->width = desc->width;
    ctx->height = desc->height;
    ctx->preset = std::min(desc->preset, 3u);
    if (desc->reserved[6] == kCreateOptionsV2Magic) {
        ctx->options.style = std::min(desc->reserved[0], 2u);
        ctx->options.use_auto_mask = (desc->reserved[1] & 1u) ? 1u : 0u;
        ctx->options.ui_correction = (desc->reserved[1] & 2u) ? 1u : 0u;
        float decoded[4]{};
        const uint32_t bits[4]{desc->reserved[2], desc->reserved[3],
                               desc->reserved[4], desc->reserved[5]};
        std::memcpy(decoded, bits, sizeof(decoded));
        if (std::isfinite(decoded[0]) && decoded[0] >= 0.0F && decoded[0] <= 2.0F)
            ctx->options.intensity = decoded[0];
        if (std::isfinite(decoded[1]) && decoded[1] >= 0.0F && decoded[1] <= 2.0F)
            ctx->options.local_tone = decoded[1];
        if (std::isfinite(decoded[2]) && decoded[2] >= 0.0F && decoded[2] <= 2.0F)
            ctx->options.local_structure = decoded[2];
        if (std::isfinite(decoded[3]) && decoded[3] >= -1.0F && decoded[3] <= 2.0F)
            ctx->options.skin_structure = decoded[3];
    }
    ctx->requested_luid = desc->adapter_luid;
    try {
        std::scoped_lock execution(g_runtime_mutex);
        if (!ctx->initialize()) return 12;
    } catch (const std::exception& e) {
        set_global_error(std::string("DLSSNR create exception: ") + e.what());
        return 13;
    } catch (...) {
        set_global_error("DLSSNR create unknown exception");
        return 14;
    }
    *context_out = ctx.release();
    return 0;
}

NEO_DLSSNR_EXPORT int32_t neo_dlssnr_process_rgba8(
    void* raw, const NeoDlssNrFrameDesc* desc, const uint8_t* input,
    uint32_t input_stride, uint8_t* output, uint32_t output_stride) {
    auto* ctx = static_cast<Context*>(raw);
    if (!ctx || !desc || !input || !output) {
        set_global_error("Null argument passed to DLSSNR evaluate");
        return 20;
    }
    try {
        std::scoped_lock execution(g_runtime_mutex);
        std::scoped_lock local(ctx->mutex);
        if (!ctx->process(desc, input, input_stride, output, output_stride)) return 21;
        return 0;
    } catch (const std::exception& e) {
        ctx->fail(std::string("DLSSNR evaluate exception: ") + e.what());
        return 22;
    } catch (...) {
        ctx->fail("DLSSNR evaluate unknown exception");
        return 23;
    }
}

NEO_DLSSNR_EXPORT int32_t neo_dlssnr_reset_history(void* raw) {
    try {
    auto* ctx = static_cast<Context*>(raw);
    if (!ctx) { set_global_error("Null DLSSNR context"); return 30; }
    std::scoped_lock lock(ctx->mutex);
    ctx->reset_pending = true;
    ctx->last_error.clear();
    return 0;
    } catch (...) { return 31; }
}

NEO_DLSSNR_EXPORT int32_t neo_dlssnr_set_options(
    void* raw, const NeoDlssNrEvalOptions* options) {
    try {
        auto* ctx = static_cast<Context*>(raw);
        if (!ctx || !options || options->struct_size < kEvalOptionsV1Size) return 40;
        if (!std::isfinite(options->intensity) || options->intensity < 0.0F || options->intensity > 2.0F ||
            !std::isfinite(options->local_tone) || options->local_tone < 0.0F || options->local_tone > 2.0F ||
            !std::isfinite(options->local_structure) || options->local_structure < 0.0F || options->local_structure > 2.0F ||
            !std::isfinite(options->skin_structure) || options->skin_structure < -1.0F || options->skin_structure > 2.0F)
            return 41;

        std::scoped_lock lock(ctx->mutex);
        NeoDlssNrEvalOptions next = ctx->options;
        next.struct_size = sizeof(NeoDlssNrEvalOptions);
        next.intensity = options->intensity;
        next.local_tone = options->local_tone;
        next.local_structure = options->local_structure;
        next.skin_structure = options->skin_structure;
        if (options->struct_size >= sizeof(NeoDlssNrEvalOptions)) {
            if (options->style > 2u || options->use_auto_mask > 1u || options->ui_correction > 1u)
                return 41;
            next.style = options->style;
            next.use_auto_mask = options->use_auto_mask;
            next.ui_correction = options->ui_correction;
            next.reserved = 0;
        }
        const bool changed =
            next.intensity != ctx->options.intensity ||
            next.local_tone != ctx->options.local_tone ||
            next.local_structure != ctx->options.local_structure ||
            next.skin_structure != ctx->options.skin_structure ||
            next.style != ctx->options.style ||
            next.use_auto_mask != ctx->options.use_auto_mask ||
            next.ui_correction != ctx->options.ui_correction;
        if (changed) {
            ctx->options = next;
            ctx->reset_pending = true;
        }
        return 0;
    } catch (...) { return 42; }
}

NEO_DLSSNR_EXPORT void neo_dlssnr_destroy(void* raw) {
    try {
    auto* ctx = static_cast<Context*>(raw);
    if (!ctx) return;
    std::scoped_lock execution(g_runtime_mutex);
    delete ctx;
    } catch (...) { }
}

NEO_DLSSNR_EXPORT uint32_t neo_dlssnr_last_error(void* raw, char* buffer,
                                                 uint32_t capacity) {
    try {
    if (auto* ctx = static_cast<Context*>(raw)) {
        std::scoped_lock lock(ctx->mutex);
        if (!ctx->last_error.empty()) return copy_string(ctx->last_error, buffer, capacity);
    }
    std::scoped_lock lock(g_error_mutex);
    return copy_string(g_last_error, buffer, capacity);
    } catch (...) {
        if (buffer && capacity) buffer[0] = '\0';
        return 0;
    }
}
