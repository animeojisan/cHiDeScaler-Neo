use std::collections::HashMap;
use std::time::{Duration, Instant};
use windows::Win32::Foundation::{CloseHandle, FILETIME};
use windows::Win32::System::Performance::{
    PDH_FMT_COUNTERVALUE_ITEM_W, PDH_FMT_DOUBLE, PDH_HCOUNTER, PDH_HQUERY, PdhAddEnglishCounterW,
    PdhCloseQuery, PdhCollectQueryData, PdhGetFormattedCounterArrayW, PdhOpenQueryW,
};
use windows::Win32::System::Threading::{
    GetSystemTimes, OpenProcess, PROCESS_NAME_WIN32, PROCESS_QUERY_LIMITED_INFORMATION,
    QueryFullProcessImageNameW,
};
use windows::core::{PCWSTR, PWSTR};

#[derive(Clone)]
struct GpuDiagItem {
    pid: u32,
    physical_key: String,
    value: f64,
}

pub struct ResourceMonitor {
    cpu: f32,
    gpu: Option<f32>,
    previous_times: Option<(u64, u64, u64)>,
    query: PDH_HQUERY,
    counter: PDH_HCOUNTER,
    last_update: Instant,
    last_gpu_diag_log: Instant,
    process_names: HashMap<u32, String>,
}

impl ResourceMonitor {
    pub fn new() -> Self {
        let mut monitor = Self {
            cpu: 0.0,
            gpu: None,
            previous_times: None,
            query: PDH_HQUERY::default(),
            counter: PDH_HCOUNTER::default(),
            last_update: Instant::now() - Duration::from_secs(2),
            last_gpu_diag_log: Instant::now() - Duration::from_secs(2),
            process_names: HashMap::new(),
        };
        monitor.init_gpu();
        monitor
    }

    pub fn sample(&mut self) -> (f32, Option<f32>) {
        if let Ok(value) = std::env::var("NEO_QA_RESOURCE_LOAD")
            && let Ok(value) = value.parse::<f32>()
        {
            let value = value.clamp(0.0, 100.0);
            return (value, Some(value));
        }
        if self.last_update.elapsed() >= Duration::from_millis(750) {
            self.last_update = Instant::now();
            self.update_cpu();
            self.update_gpu();
        }
        (self.cpu, self.gpu)
    }

    fn init_gpu(&mut self) {
        let path: Vec<u16> = "\\GPU Engine(*)\\Utilization Percentage\0"
            .encode_utf16()
            .collect();
        unsafe {
            if PdhOpenQueryW(PCWSTR::null(), 0, &mut self.query) != 0 {
                return;
            }
            if PdhAddEnglishCounterW(self.query, PCWSTR(path.as_ptr()), 0, &mut self.counter) != 0 {
                let _ = PdhCloseQuery(self.query);
                self.query = PDH_HQUERY::default();
                return;
            }
            let _ = PdhCollectQueryData(self.query);
        }
    }

    fn update_cpu(&mut self) {
        let mut idle = FILETIME::default();
        let mut kernel = FILETIME::default();
        let mut user = FILETIME::default();
        if unsafe { GetSystemTimes(Some(&mut idle), Some(&mut kernel), Some(&mut user)) }.is_err() {
            return;
        }
        let current = (filetime(idle), filetime(kernel), filetime(user));
        if let Some(previous) = self.previous_times {
            let idle_delta = current.0.saturating_sub(previous.0);
            let total = current.1.saturating_sub(previous.1) + current.2.saturating_sub(previous.2);
            if total > 0 {
                self.cpu = ((total.saturating_sub(idle_delta)) as f64 * 100.0 / total as f64)
                    .clamp(0.0, 100.0) as f32;
            }
        }
        self.previous_times = Some(current);
    }

    fn update_gpu(&mut self) {
        if self.query == PDH_HQUERY::default() || self.counter == PDH_HCOUNTER::default() {
            return;
        }
        unsafe {
            if PdhCollectQueryData(self.query) != 0 {
                return;
            }
            let mut bytes = 0u32;
            let mut count = 0u32;
            const PDH_MORE_DATA: u32 = 0x8000_07d2;
            if PdhGetFormattedCounterArrayW(
                self.counter,
                PDH_FMT_DOUBLE,
                &mut bytes,
                &mut count,
                None,
            ) != PDH_MORE_DATA
                || bytes == 0
            {
                return;
            }
            let mut buffer = vec![0u8; bytes as usize];
            let items = buffer.as_mut_ptr().cast::<PDH_FMT_COUNTERVALUE_ITEM_W>();
            if PdhGetFormattedCounterArrayW(
                self.counter,
                PDH_FMT_DOUBLE,
                &mut bytes,
                &mut count,
                Some(items),
            ) != 0
            {
                return;
            }

            // IMPORTANT: `engines` is intentionally the exact legacy GUI-meter
            // calculation. Do not change it while diagnosing the 25% issue.
            let mut engines: HashMap<String, f64> = HashMap::new();
            // Diagnostic-only grouping keeps the LUID/physical adapter in the
            // key so a multi-GPU machine cannot hide which real GPU engine is
            // busy. This does NOT feed the GUI value.
            let mut physical_engines: HashMap<String, f64> = HashMap::new();
            let mut per_pid_engine: HashMap<(u32, String), f64> = HashMap::new();
            let mut diag_items = Vec::<GpuDiagItem>::new();

            for item in std::slice::from_raw_parts(items, count as usize) {
                if item.FmtValue.CStatus != 0 {
                    continue;
                }
                let value = item.FmtValue.Anonymous.doubleValue;
                if value <= 0.0 || item.szName.is_null() {
                    continue;
                }
                let name = item.szName.to_string().unwrap_or_default();
                let legacy_key = name
                    .find("_eng_")
                    .map(|index| name[index + 1..].to_owned())
                    .unwrap_or_else(|| name.clone());
                *engines.entry(legacy_key).or_default() += value;

                let physical_key = physical_engine_key(&name);
                *physical_engines.entry(physical_key.clone()).or_default() += value;
                let pid = parse_pid(&name).unwrap_or(0);
                if pid != 0 {
                    *per_pid_engine
                        .entry((pid, physical_key.clone()))
                        .or_default() += value;
                }
                diag_items.push(GpuDiagItem {
                    pid,
                    physical_key,
                    value,
                });
            }

            let (legacy_key, legacy_value) = top_entry(&engines)
                .map(|(key, value)| (key.clone(), *value))
                .unwrap_or_else(|| ("none".to_string(), 0.0));
            self.gpu = Some(legacy_value.clamp(0.0, 100.0) as f32);

            // Diagnostic logging only. Keep enough raw Windows GPU-engine
            // evidence in the normal log to correlate the displayed percentage
            // with individual engines and processes without external tools.
            if log::log_enabled!(log::Level::Info)
                && self.last_gpu_diag_log.elapsed() >= Duration::from_millis(1500)
            {
                self.last_gpu_diag_log = Instant::now();
                let (physical_key, physical_value) = top_entry(&physical_engines)
                    .map(|(key, value)| (key.clone(), *value))
                    .unwrap_or_else(|| ("none".to_string(), 0.0));

                let mut physical_top: Vec<_> = physical_engines
                    .iter()
                    .map(|(key, value)| (key.clone(), *value))
                    .collect();
                physical_top.sort_by(|a, b| b.1.total_cmp(&a.1));
                physical_top.truncate(4);
                let physical_top_text = physical_top
                    .iter()
                    .map(|(key, value)| format!("{}={:.1}%", compact_engine_key(key), value))
                    .collect::<Vec<_>>()
                    .join(",");

                let mut contributors: Vec<_> = diag_items
                    .iter()
                    .filter(|item| item.physical_key == physical_key && item.pid != 0)
                    .map(|item| (item.pid, item.value))
                    .collect();
                contributors.sort_by(|a, b| b.1.total_cmp(&a.1));
                contributors.truncate(8);
                let contributors_text = contributors
                    .into_iter()
                    .map(|(pid, value)| {
                        let name = self.process_name(pid);
                        format!("{}({})={:.1}%", name, pid, value)
                    })
                    .collect::<Vec<_>>()
                    .join(",");

                let mut process_max: HashMap<u32, f64> = HashMap::new();
                for ((pid, _engine), value) in &per_pid_engine {
                    process_max
                        .entry(*pid)
                        .and_modify(|current| *current = current.max(*value))
                        .or_insert(*value);
                }
                let mut process_top: Vec<_> = process_max.into_iter().collect();
                process_top.sort_by(|a, b| b.1.total_cmp(&a.1));
                process_top.truncate(8);
                let process_top_text = process_top
                    .into_iter()
                    .map(|(pid, value)| {
                        let name = self.process_name(pid);
                        format!("{}({})={:.1}%", name, pid, value)
                    })
                    .collect::<Vec<_>>()
                    .join(",");

                log::info!(
                    "gpu-diag: gui_legacy={:.1}% legacy_engine='{}' physical_top={:.1}% physical_engine='{}' top_engines=[{}] contributors=[{}] process_max=[{}]",
                    legacy_value.clamp(0.0, 100.0),
                    compact_engine_key(&legacy_key),
                    physical_value.clamp(0.0, 100.0),
                    compact_engine_key(&physical_key),
                    physical_top_text,
                    contributors_text,
                    process_top_text,
                );
            }
        }
    }

    fn process_name(&mut self, pid: u32) -> String {
        if let Some(name) = self.process_names.get(&pid) {
            return name.clone();
        }
        let name = query_process_name(pid).unwrap_or_else(|| format!("pid-{pid}"));
        self.process_names.insert(pid, name.clone());
        name
    }
}

impl Drop for ResourceMonitor {
    fn drop(&mut self) {
        if self.query != PDH_HQUERY::default() {
            unsafe {
                let _ = PdhCloseQuery(self.query);
            }
        }
    }
}

fn top_entry(map: &HashMap<String, f64>) -> Option<(&String, &f64)> {
    map.iter().max_by(|a, b| a.1.total_cmp(b.1))
}

fn parse_pid(name: &str) -> Option<u32> {
    let rest = name.strip_prefix("pid_")?;
    let end = rest.find('_').unwrap_or(rest.len());
    rest[..end].parse().ok()
}

fn physical_engine_key(name: &str) -> String {
    // PDH GPU Engine instances are normally:
    // pid_N_luid_..._phys_N_eng_N_engtype_3D
    // Strip only the pid prefix so processes sharing one *physical* engine
    // aggregate together while separate adapters/LUIDs remain separate.
    name.find("_luid_")
        .map(|index| name[index + 1..].to_owned())
        .unwrap_or_else(|| name.to_owned())
}

fn compact_engine_key(key: &str) -> String {
    // Keep enough identity to compare adapters and engine type, but avoid
    // dumping very long PDH instance strings into every line.
    let eng = key
        .find("_eng_")
        .map(|index| &key[index + 1..])
        .unwrap_or(key);
    let phys = token_value(key, "phys_").unwrap_or("?");
    let luid = key
        .find("luid_")
        .and_then(|start| {
            key[start + 5..]
                .find("_phys_")
                .map(|end| &key[start + 5..start + 5 + end])
        })
        .unwrap_or("?");
    format!("luid={luid}/phys={phys}/{eng}")
}

fn token_value<'a>(text: &'a str, token: &str) -> Option<&'a str> {
    let start = text.find(token)? + token.len();
    let rest = &text[start..];
    let end = rest.find('_').unwrap_or(rest.len());
    Some(&rest[..end])
}

fn query_process_name(pid: u32) -> Option<String> {
    unsafe {
        let process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid).ok()?;
        let mut path = [0u16; 1024];
        let mut len = path.len() as u32;
        let ok = QueryFullProcessImageNameW(
            process,
            PROCESS_NAME_WIN32,
            PWSTR(path.as_mut_ptr()),
            &mut len,
        )
        .is_ok();
        let _ = CloseHandle(process);
        if !ok || len == 0 {
            return None;
        }
        let path = String::from_utf16_lossy(&path[..len as usize]);
        Some(
            path.rsplit(['\\', '/'])
                .next()
                .filter(|name| !name.is_empty())
                .unwrap_or(&path)
                .to_string(),
        )
    }
}

fn filetime(value: FILETIME) -> u64 {
    (u64::from(value.dwHighDateTime) << 32) | u64::from(value.dwLowDateTime)
}
