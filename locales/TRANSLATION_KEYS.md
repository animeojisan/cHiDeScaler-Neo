# Translation keys

The English catalog is authoritative.

Every locale JSON must start with `_language_name`, `_language_code`, and `_language_tag`. These are Language UI metadata, not ordinary translation keys; see [`README.md`](README.md) or [`README_ja.md`](README_ja.md).

| Key | English |
|---|---|
| `admin.dialog_title` | Administrator permission required |
| `admin.required_help` | Required when controlling windows that are running as administrator |
| `admin.restart` | Restart as administrator |
| `admin.restart_failed` | Could not restart as administrator. Allow the Windows confirmation prompt and try again. |
| `admin.restart_instruction` | Restart cHiDeScaler-Neo as administrator, then select the window again. |
| `admin.target_elevated` | The selected window is running with administrator permission. |
| `capture.click_target` | (Click the window to magnify) |
| `capture.client_help` | Capture only the window contents; applies next start |
| `capture.client_forced_help` | Capture resolution forces client-area-only capture until Auto is selected |
| `capture.resolution_failed` | The source window cannot keep the requested capture resolution. Select Auto or a supported size. Capture was not started. |
| `capture.running` | ● Running |
| `capture.short` | Capture: |
| `capture.size` | Capture size: |
| `capture.size_help` | Auto keeps the current source size. A preset/custom value resizes the source client area. |
| `capture.size_select_help` | Select the capture resolution. Auto keeps the current source size. |
| `capture.fullscreen_notice_title` | Capture resolution |
| `capture.fullscreen_notice_body` | Capture resolution is temporarily disabled while the source is fullscreen. Neo follows the source resolution reported by WGC. Your selected capture resolution will be applied again automatically after the source returns to windowed mode. |
| `capture.start` | ▶ Start |
| `capture.preparing` | Preparing… |
| `capture.stopping` | Stopping… |
| `capture.stop` | Stop |
| `capture.stop_icon` | ■ Stop |
| `capture.target` | 🎯 Target: |
| `common.apply` | Apply |
| `common.auto` | Auto |
| `common.builtin` | Built-in |
| `common.cancel` | Cancel |
| `common.delete` | Delete |
| `common.edit` | Edit |
| `common.move_down` | Move down |
| `common.move_up` | Move up |
| `common.new` | New |
| `common.overwrite` | Overwrite |
| `common.quit` | Quit |
| `common.save` | Save |
| `common.seconds` |  sec |
| `cursor.speed_help` | Make the cursor feel like normal desktop speed over the magnified view |
| `display.fullscreen` | Fullscreen |
| `display.label` | Display: |
| `display.windowed` | Windowed |
| `duplicate.help` | Skips confidently identical frames and presents the completed output with the selected timing. |
| `duplicate.summary` | Reuses the previous completed output for confidently identical frames while preserving presentation timing. |
| `filter.add` | Add Filter |
| `filter.add_icon` | + Add Filter |
| `filter.chain` | Filter Chain |
| `filter.empty_help` | No filters in this chain. Use + Add Filter to add one. |
| `filter.more_errors` | more filter errors |
| `filter.none` | (No filters found) |
| `filter.reorder_help` | (Hold and drag to reorder) |
| `filter.toggle` | Enable/Disable |
| `glsl_overload.lighter_filter_help` | Switch to a lighter filter to clear this warning. |
| `glsl_overload.pause` | The filter is overloaded. Processing is paused while you use the GUI to protect responsiveness. |
| `hdr.help` | Reduces blown highlights in HDR video. Applies next start. |
| `hotkey.edit_help` | Hold Ctrl, Alt, or Shift and press a letter, number, function key, or navigation key. |
| `hotkey.error.conflict` | This shortcut is already used by Windows or another application. |
| `hotkey.error.duplicate` | The same key is duplicated. |
| `hotkey.error.key_count` | Use a total of two or three keys, including modifiers. |
| `hotkey.error.modifier_required` | Ctrl, Alt, or Shift is required. |
| `hotkey.error.one_primary` | Only one non-modifier key is allowed. |
| `hotkey.error.primary_required` | Modifier-only shortcuts cannot be registered. Add a letter, number, or function key. |
| `hotkey.error.registration` | The shortcut became unavailable during registration. The previous shortcut was restored. |
| `hotkey.error.reserved` | This combination is reserved by the app or Windows. |
| `hotkey.error.unsupported` | This key is not supported for global shortcuts. |
| `hotkey.error.win_blocked` | Win-key combinations are blocked because they conflict with Windows shortcuts. |
| `hotkey.rules` | Two or three keys total. Modifier-only, Win-key, and reserved shortcuts are blocked. |
| `interpolation.factor_help` | Output multiplier for frame-interpolation filters. x4/x5 target high-refresh displays and are limited automatically by the monitor refresh rate. |
| `log.help` | Write bounded rotating cHiDeScaler-Neo.log diagnostics for troubleshooting |
| `panel.gui_topmost_disable_help` | Stop keeping the Neo GUI always on top. |
| `panel.gui_topmost_enable_help` | Keep the Neo GUI always on top. |
| `panel.keep_visible` | Keep visible |
| `panel.minimize_help` | Minimize panel (hover to restore) |
| `panel.screenshot` | Screenshot |
| `panel.show_help` | Show the floating stop/collapse panel during capture |
| `preset.delete` | Delete Preset |
| `preset.delete_confirm` | Delete preset '{name}'? |
| `preset.enter_name` | Enter a preset name. |
| `preset.exists` | A preset with this name already exists. |
| `preset.new` | New Preset |
| `preset.not_found` | The selected preset could not be found. |
| `preset.overwrite_help` | Overwrite this preset with the current chain |
| `preset.reorder_help` | Long press and drag to reorder |
| `preset.save_as` | Save As |
| `resize.edit` | Edit resize scale |
| `resize.final_help` | Final resize filter used to fit the chain output to the window/monitor (upscale or downscale) |
| `resize.range` | 0.25–4.00 (default 0.75) |
| `resize.scale` | Scale |
| `resize.title` | Resize scale |
| `settings.client_only` | Client area only |
| `settings.cursor_autohide` | Auto-hide cursor |
| `settings.cursor_speed` | Natural cursor speed |
| `settings.duplicate_reduction` | Duplicate reduction |
| `settings.fps_cap` | FPS cap |
| `settings.frame_interpolation` | Frame interpolation: |
| `settings.gui_topmost` | Keep GUI on top |
| `settings.hdr_sdr` | HDR highlight protection |
| `settings.open_folder` | Open settings folder |
| `settings.open_folder_failed` | Could not open the settings folder |
| `settings.panel_show` | Show control panel |
| `settings.resize` | Resize: |
| `settings.restart_admin` | Restart as admin |
| `settings.save_log` | Save log |
| `settings.smooth_pacing` | Smooth pacing |
| `settings.stats` | Stats |
| `settings.vsync` | VSync |
| `shortcut.edit` | Edit Shortcut |
| `shortcut.gui` | GUI |
| `shortcut.panel` | Panel |
| `shortcut.start_stop` | Start/Stop |
| `shortcut.title` | Shortcuts |
| `smooth.help` | Stabilizes presentation timing to prioritize smooth scrolling and motion. Separate from VSync; if tearing or split-frame artifacts are noticeable, try VSync as well. |
| `stats.help` | Total is the measured time from capture arrival to presentation.<br>Delay frames are total ms divided by the source frame interval. |
| `stats.main` | Input: {in_w}x{in_h} → Internal: {internal_w}x{internal_h} → Output: {out_w}x{out_h}   Total: {total_ms}ms / Delay: {lag_frames} frames   Present: {present_fps}fps / Capture: {capture_fps}fps |
| `stats.monitor` | Monitor: {width}x{height} @ {refresh}Hz |
| `tensorrt.cuda_fallback_count` | CUDA fallback: {count} filter(s) |
| `tensorrt.directml_fallback_count` | DirectML fallback: {count} filter(s) |
| `tensorrt.help` | Run ONNX filters with TensorRT priority on an NVIDIA GPU. Unsupported filters automatically fall back to DirectML. |
| `tensorrt.pack_required` | The separately distributed TensorRT Backend Pack is required. |
| `tensorrt.preparing_label` | TensorRT (preparing) |
| `tensorrt.unavailable_help` | The TensorRT backend is currently unavailable. Check cHiDeScaler-Neo.log for details. |
| `vsync.help` | Synchronizes presentation to the monitor refresh to reduce tearing or split-frame artifacts; scrolling may feel less smooth on some systems. |
