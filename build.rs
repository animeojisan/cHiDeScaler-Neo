fn main() {
    guard_text_encoding();
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
            // Hybrid-GPU drivers inspect these exported data symbols before
            // creating the first WGL context. Keep native WGL/GLSL while
            // requesting the discrete/high-performance adapter.
            println!("cargo:rustc-link-arg-bin=chidescaler-neo=/EXPORT:NvOptimusEnablement,DATA");
            println!(
                "cargo:rustc-link-arg-bin=chidescaler-neo=/EXPORT:AmdPowerXpressRequestHighPerformance,DATA"
            );
        }
        let mut res = winresource::WindowsResource::new();
        res.set_icon("assets/cHiDeScaler-Neo.ico");
        // Populate standard VERSIONINFO fields for consistent Windows metadata.
        res.set("ProductName", "cHiDeScaler-Neo");
        res.set(
            "FileDescription",
            "cHiDeScaler-Neo realtime window upscaler / frame interpolator",
        );
        res.set("CompanyName", "cHiDeScaler-Neo project");
        res.set("LegalCopyright", "(c) 2026 cHiDeScaler-Neo project");
        res.set("OriginalFilename", "cHiDeScaler-Neo.exe");
        res.set("InternalName", "chidescaler-neo");
        res.set("ProductVersion", "0.99.2.0");
        res.set("FileVersion", "0.99.2.0");
        res.set_version_info(
            winresource::VersionInfo::PRODUCTVERSION,
            0x0000_0063_0002_0000,
        );
        res.set_version_info(winresource::VersionInfo::FILEVERSION, 0x0000_0063_0002_0000);
        if let Err(e) = res.compile() {
            println!("cargo:warning=icon resource compile failed: {e}");
        }
    }
}

fn guard_text_encoding() {
    let root = std::path::PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string()),
    );
    let mut files = Vec::new();
    collect_text_files(&root.join("src"), &mut files);
    if let Ok(entries) = std::fs::read_dir(&root) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_file() && is_checked_text_file(&p) {
                files.push(p);
            }
        }
    }
    for file in files {
        println!("cargo:rerun-if-changed={}", file.display());
        let text = std::fs::read_to_string(&file)
            .unwrap_or_else(|e| panic!("{} is not valid UTF-8: {e}", file.display()));
        if let Some(ch) = text.chars().find(|c| mojibake_marker(*c)) {
            panic!(
                "{} contains a mojibake/replacement marker U+{:04X}; save text files as UTF-8",
                file.display(),
                ch as u32
            );
        }
    }
}

fn collect_text_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            collect_text_files(&p, out);
        } else if is_checked_text_file(&p) {
            out.push(p);
        }
    }
}

fn is_checked_text_file(path: &std::path::Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()),
        Some("rs" | "toml" | "md" | "txt" | "ps1")
    )
}

fn mojibake_marker(ch: char) -> bool {
    matches!(
        ch,
        '\u{7e3a}'
            | '\u{7e67}'
            | '\u{7e5d}'
            | '\u{8b41}'
            | '\u{8b5b}'
            | '\u{8b4c}'
            | '\u{90b1}'
            | '\u{8708}'
            | '\u{8703}'
            | '\u{879f}'
            | '\u{9082}'
            | '\u{8373}'
            | '\u{7aca}'
            | '\u{7b06}'
            | '\u{7ab6}'
            | '\u{8b56}'
            | '\u{8b20}'
            | '\u{870d}'
            | '\u{8711}'
            | '\u{970e}'
            | '\u{f8f0}'
            | '\u{fffd}'
    )
}
