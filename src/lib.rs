pub mod engine;
pub mod i18n;
pub mod input;
pub mod logging;
pub mod capture {
    pub mod wgc;
}
pub mod core {
    pub mod config;
    pub mod dlssnr;
    pub mod metrics;
    pub mod presets;
}
pub mod overlay {
    pub mod window;
}
pub mod platform {
    pub mod gl_window;
    pub mod gpu;
    pub mod hotkeys;
    pub mod startup;
    pub mod tray;
    pub mod win32;
}
pub mod render {
    pub mod chain;
    pub mod cuda_interop;
    pub mod dlssnr_backend;
    pub mod dlssnr_stage;
    pub mod flow;
    pub mod flow_kari;
    pub mod flow_sigma;
    pub mod gl;
    pub mod glsl_engine;
    pub mod mpv;
    pub mod onnx_accel;
    pub mod onnx_backend;
    pub mod neoamd_backend;
    pub mod onnx_stage;
    pub mod scaler;
    pub mod slangp;
    pub mod vulkan_gpu;
    pub mod vulkan_multipass;
    pub mod vulkan_onepass;
    pub mod winml_migraphx;
}
