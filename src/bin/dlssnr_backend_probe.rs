//! Developer-only DLSSNR Backend Pack ABI smoke probe.
//!
//! It is never started by the normal Neo executable. Run explicitly on Windows:
//!   cargo run --bin dlssnr_backend_probe -- <Neo app directory>

use chidescaler_neo::core::dlssnr::DlssNrOptions;
use chidescaler_neo::render::dlssnr_backend::{
    DLSSNR_CAP_ADVANCED_OPTIONS, DLSSNR_CAP_ZERO_GUIDANCE, DlssNrBackend,
    detect_dlssnr_backend_pack,
};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

fn main() {
    let app_dir = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().expect("current directory"));
    let availability = detect_dlssnr_backend_pack(&app_dir);
    if !availability.installed {
        eprintln!(
            "DLSSNR pack unavailable: {}",
            availability.reason.as_deref().unwrap_or("unknown reason")
        );
        std::process::exit(2);
    }
    let backend = match DlssNrBackend::load_verified(&availability) {
        Ok(backend) => backend,
        Err(error) => {
            eprintln!("DLSSNR bridge load failed: {error}");
            std::process::exit(3);
        }
    };

    if let Some(manifest) = availability.manifest.as_ref() {
        println!("Backend pack: {}", manifest.name);
        if !manifest.compatibility_tags.is_empty() {
            println!(
                "Compatibility tags: {}",
                manifest.compatibility_tags.join(", ")
            );
        }
        if !manifest.runtime_candidates.is_empty() {
            println!(
                "Runtime candidates: {}",
                manifest.runtime_candidates.join(", ")
            );
        }
        if manifest.allow_user_runtime_replacement {
            println!(
                "User runtime replacement: enabled ({})",
                manifest.replaceable_runtime_files.join(", ")
            );
        } else {
            println!("User runtime replacement: disabled");
        }
    }
    let capabilities = backend.capabilities();
    println!("Bridge capabilities: 0x{capabilities:016X}");
    if capabilities & DLSSNR_CAP_ZERO_GUIDANCE == 0 {
        eprintln!(
            "DLSSNR bridge loaded but does not advertise Zero Guidance; refusing to treat a pass-through ABI stub as a Feature 18 test"
        );
        std::process::exit(6);
    }
    if let Some(description) = backend.backend_description() {
        println!("Bridge description: {description}");
    }

    // Use a realistic default geometry: private Feature 18 runtimes may reject
    // very small textures. Override with NEO_DLSSNR_PROBE=WxH when needed.
    let (mut width, mut height) = std::env::var("NEO_DLSSNR_PROBE")
        .ok()
        .and_then(|value| {
            value
                .split_once('x')
                .map(|(w, h)| (w.to_owned(), h.to_owned()))
        })
        .and_then(|(w, h)| Some((w.parse::<u32>().ok()?, h.parse::<u32>().ok()?)))
        .unwrap_or((1280, 720));
    let mut stride = width * 4;
    let mut input = vec![0u8; stride as usize * height as usize];
    for y in 0..height {
        for x in 0..width {
            let i = (y * stride + x * 4) as usize;
            input[i] = (x & 0xff) as u8;
            input[i + 1] = (y & 0xff) as u8;
            input[i + 2] = ((x ^ y) & 0xff) as u8;
            input[i + 3] = 255;
        }
    }

    if let Some(path) = std::env::var_os("NEO_DLSSNR_PROBE_IMAGE") {
        let mut decoder =
            png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path).unwrap()));
        decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
        let mut reader = decoder.read_info().unwrap();
        let mut bytes = vec![0; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut bytes).unwrap();
        width = info.width;
        height = info.height;
        stride = width * 4;
        input = match info.color_type {
            png::ColorType::Rgba => bytes[..info.buffer_size()].to_vec(),
            png::ColorType::Rgb => bytes[..info.buffer_size()]
                .chunks_exact(3)
                .flat_map(|p| [p[0], p[1], p[2], 255])
                .collect(),
            _ => panic!("probe image must be RGB/RGBA"),
        };
    }

    let mut session = match backend.create_session(None, width, height, 1) {
        Ok(session) => session,
        Err(error) => {
            eprintln!("DLSSNR session creation failed: {error}");
            std::process::exit(4);
        }
    };
    if let Some(dir) = std::env::var_os("NEO_DLSSNR_PROBE_OPTIONS") {
        let dir = PathBuf::from(dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut cases = vec![("baseline".to_string(), DlssNrOptions::default())];
        for (key, field) in [
            ("intensity", 0usize),
            ("local_tone", 1),
            ("local_structure", 2),
            ("skin_structure", 3),
        ] {
            for value in [0.0, 0.5] {
                let mut values = DlssNrOptions::default();
                match field {
                    0 => values.intensity = value,
                    1 => values.local_tone = value,
                    2 => values.local_structure = value,
                    _ => values.skin_structure = value,
                }
                cases.push((format!("{key}_{value}"), values));
            }
        }
        if capabilities & DLSSNR_CAP_ADVANCED_OPTIONS != 0 {
            let mut strong = DlssNrOptions::default();
            strong.intensity = 2.0;
            strong.local_structure = 2.0;
            cases.push(("advanced_strength_2".to_string(), strong));
        }
        cases.push(("baseline_repeat".to_string(), DlssNrOptions::default()));
        let mut baseline = Vec::new();
        for (name, values) in cases {
            session.set_options(values).expect("set options");
            let mut result = Vec::new();
            let started = std::time::Instant::now();
            for _ in 0..20 {
                result = session
                    .process_rgba8(&input, stride, true)
                    .expect("evaluate options");
            }
            if baseline.is_empty() {
                baseline = result.clone();
            }
            let differences: Vec<_> = result
                .chunks_exact(4)
                .zip(baseline.chunks_exact(4))
                .flat_map(|(a, b)| (0..3).map(move |i| (a[i] as f64 - b[i] as f64).abs()))
                .collect();
            let mean = differences.iter().sum::<f64>() / differences.len() as f64;
            let max = differences.iter().copied().fold(0.0, f64::max);
            let changed = differences.iter().filter(|v| **v != 0.0).count();
            println!(
                "OPTIONS case={name} values={values:?} mae_8bit={mean:.6} max={max} changed={changed} mean_ms={:.3} sha256={:X}",
                started.elapsed().as_secs_f64() * 50.0,
                Sha256::digest(&result)
            );
            let mut encoder = png::Encoder::new(
                std::io::BufWriter::new(
                    std::fs::File::create(dir.join(format!("{name}.png"))).unwrap(),
                ),
                width,
                height,
            );
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            encoder
                .write_header()
                .unwrap()
                .write_image_data(&result)
                .unwrap();
        }
        return;
    }
    let output = match session.process_rgba8(&input, stride, true) {
        Ok(output) => output,
        Err(error) => {
            eprintln!("DLSSNR evaluation failed: {error}");
            std::process::exit(5);
        }
    };
    let hash = Sha256::digest(&output);
    let changed_bytes = input
        .iter()
        .zip(output.iter())
        .filter(|(a, b)| a != b)
        .count();
    println!(
        "DLSSNR Feature 18 smoke succeeded: {}x{} output_bytes={} changed_bytes={} sha256={:X}",
        width,
        height,
        output.len(),
        changed_bytes,
        hash
    );
}
