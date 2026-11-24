//! Sample the GPU timeline without fences, waits, or UI statistics rows.
use glow::HasContext;
use std::{collections::VecDeque, rc::Rc, time::Instant};

struct Pair(glow::Query, glow::Query);
struct Sample {
    id: u64,
    seq: u64,
    suffix: usize,
    pair: Pair,
    started: Instant,
    host_ms: f64,
    back_buffer: bool,
    ready_at_swap: Option<bool>,
}

pub(crate) struct AsyncInterpTiming {
    gl: Rc<glow::Context>,
    enabled: bool,
    ordinal: u64,
    every: u64,
    free: Vec<Pair>,
    active: Option<Sample>,
    pending: VecDeque<Sample>,
}

impl AsyncInterpTiming {
    pub(crate) fn new(gl: Rc<glow::Context>) -> Self {
        let enabled = gl.supported_extensions().contains("GL_ARB_timer_query");
        Self { gl, enabled, ordinal: 0, every: 16, free: Vec::new(), active: None, pending: VecDeque::new() }
    }

    pub(crate) fn for_probe(gl: Rc<glow::Context>) -> Self {
        let mut result=Self::new(gl);result.every=1;result
    }

    fn delete(&self, pair: Pair) {
        unsafe { self.gl.delete_query(pair.0); self.gl.delete_query(pair.1); }
    }

    pub(crate) fn begin(&mut self, seq: u64, suffix: usize) -> Option<u64> {
        // A failed prior render may have returned before end().
        if let Some(old) = self.active.take() { self.delete(old.pair); }
        self.ordinal = self.ordinal.wrapping_add(1);
        if !self.enabled || self.ordinal % self.every != 0 || self.pending.len() >= 8 { return None; }
        let pair = if let Some(pair) = self.free.pop() { pair } else {
            unsafe {
                let first = match self.gl.create_query() { Ok(q) => q, Err(_) => { self.enabled = false; return None; } };
                let second = match self.gl.create_query() { Ok(q) => q, Err(_) => {
                    self.gl.delete_query(first); self.enabled = false; return None;
                } };
                Pair(first, second)
            }
        };
        unsafe { self.gl.query_counter(pair.0, glow::TIMESTAMP); }
        let id = self.ordinal;
        self.active = Some(Sample { id, seq, suffix, pair, started: Instant::now(), host_ms: 0.0,
            back_buffer: false, ready_at_swap: None });
        Some(id)
    }

    pub(crate) fn end(&mut self, id: u64, back_buffer: bool) {
        if !self.active.as_ref().is_some_and(|sample| sample.id == id) { return; }
        let mut sample = self.active.take().unwrap();
        unsafe { self.gl.query_counter(sample.pair.1, glow::TIMESTAMP); }
        sample.host_ms = sample.started.elapsed().as_secs_f64() * 1000.0;
        sample.back_buffer = back_buffer;
        self.pending.push_back(sample);
    }

    pub(crate) fn observe_swap(&mut self, id: u64) {
        if let Some(sample) = self.pending.iter_mut().find(|sample| sample.id == id) {
            sample.ready_at_swap = Some(unsafe {
                self.gl.get_query_parameter_u32(sample.pair.1, glow::QUERY_RESULT_AVAILABLE) != 0
            });
        }
    }

    pub(crate) fn poll(&mut self) -> usize {
        let mut collected = 0;
        while let Some(sample) = self.pending.front() {
            let available = unsafe { self.gl.get_query_parameter_u32(sample.pair.1, glow::QUERY_RESULT_AVAILABLE) != 0 };
            if !available { break; }
            let sample = self.pending.pop_front().unwrap();
            let (start, end) = unsafe {
                (self.gl.get_query_parameter_u64(sample.pair.0, glow::QUERY_RESULT),
                 self.gl.get_query_parameter_u64(sample.pair.1, glow::QUERY_RESULT))
            };
            if end >= start {
                log::debug!("interp-render-gpu-sample: ordinal={} seq={} suffix_start={} gpu_span_ms={:.3} host_submit_ms={:.3} result_age_ms={:.3} back_buffer_drawn={} ready_before_swap={:?} timing=gpu-timestamp-span-includes-queue-and-host-gaps wait_added=false",
                    sample.id, sample.seq, sample.suffix, (end-start) as f64 / 1_000_000.0, sample.host_ms,
                    sample.started.elapsed().as_secs_f64()*1000.0, sample.back_buffer, sample.ready_at_swap);
            }
            self.free.push(sample.pair);
            collected += 1;
        }
        collected
    }
}

impl Drop for AsyncInterpTiming {
    fn drop(&mut self) {
        if let Some(sample) = self.active.take() { self.delete(sample.pair); }
        while let Some(sample) = self.pending.pop_front() { self.delete(sample.pair); }
        while let Some(pair) = self.free.pop() { self.delete(pair); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sampled_timestamps_recycle_without_touching_the_framebuffer() {
        let window = crate::platform::gl_window::GlWindow::hidden(64,48).unwrap();
        let gl = Rc::new(unsafe { glow::Context::from_loader_function(|name|window.loader(name)) });
        let mut timing = AsyncInterpTiming::new(gl.clone());
        assert!(timing.enabled, "local GPU timer-query extension unavailable");
        timing.every = 1;
        for seq in 0..64 {
            let id = timing.begin(seq, 1).unwrap();
            unsafe { gl.clear_color(0.25,0.5,0.75,1.0); gl.clear(glow::COLOR_BUFFER_BIT); }
            timing.end(id, true);
            // Test synchronization only: production never calls finish here.
            unsafe { gl.finish(); }
            timing.observe_swap(id);
            assert_eq!(timing.pending.front().unwrap().ready_at_swap, Some(true));
            assert_eq!(timing.poll(), 1);
            assert_eq!(timing.free.len(), 1);
            assert!(timing.pending.is_empty());
        }
        let abandoned = timing.begin(65,1).unwrap();
        assert!(abandoned>0);
        timing.begin(66,1).unwrap(); // interrupted sample is discarded safely
        assert_eq!(timing.pending.len(),0);
    }
}
