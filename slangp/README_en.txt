cHiDeScaler-Neo External SLANGP Preset Setup Guide

Neo's SLANGP feature loads user-supplied Libretro / RetroArch Slang shaders
as external files.
Neo does not bundle RetroCrisis, Guest Advanced, Libretro shader code,
presets, LUT images, or other third-party shader data.
External shader packs remain subject to the licenses of their respective distributors.

============================================================
1. Getting and installing the required files
============================================================

[Option A: Download the official Libretro slang-shaders package from GitHub]

1. Open the official Libretro slang-shaders repository:
   https://github.com/libretro/slang-shaders

2. Select "Code" -> "Download ZIP".

3. Extract the ZIP, then rename the extracted folder
   slang-shaders-master
   to
   shaders_slang

4. Place the shaders_slang folder inside Neo's slangp folder.

Destination:
  cHiDeScaler-Neo/slangp/shaders_slang/


[Option B: Get the shaders through RetroArch]

If RetroArch is already installed, you can obtain them as follows:

1. In RetroArch, run:
   Main Menu -> Online Updater -> Update Slang Shaders

2. Copy RetroArch's
   shaders/shaders_slang
   folder.

3. Place the copied shaders_slang folder at:
   cHiDeScaler-Neo/slangp/shaders_slang/

This shader update method is also documented in the official Libretro shader guide:
   https://docs.libretro.com/guides/shaders/


============================================================
2. RetroCrisis preset download
============================================================

Retro Crisis GDV-NTSC presets are available here:

RetroCrisis / Retro-Crisis-GDV-NTSC
https://github.com/RetroCrisis/Retro-Crisis-GDV-NTSC

Use "Code" -> "Download ZIP" on GitHub, extract the archive, and place the
preset folder under Neo's slangp directory.

Example:
  cHiDeScaler-Neo/
  └─ slangp/
     ├─ shaders_slang/
     │  ├─ crt/
     │  ├─ ntsc/
     │  └─ ...
     │
     └─ retro crisis/
        ├─ 1080p Flat/
        ├─ 1080p Curved/
        ├─ 1440p Flat/
        ├─ 4K Flat/
        └─ ...

Some RetroCrisis presets reference other presets, .slang files inside the
shaders_slang tree, LUT images, and other dependencies using relative paths.
For that reason, do not copy only one .slangp file. Keep the original preset
pack directory structure intact whenever possible.

============================================================
3. Using SLANGP presets in Neo
============================================================

1. Place the required files under the slangp folder.
2. Start Neo.
3. Open the Add Filter window.
4. Select the .slangp preset you want to use.

Only .slangp preset entry points are shown in Neo's filter picker.
Referenced .slang files and LUT images are treated as dependencies and are
not listed as separate filters.

In the filter chain, each SLANGP preset is shown as a single SLANGP filter item.

============================================================
4. Portable layout
============================================================

Neo can rebase shaders_slang references used by external presets to:
  slangp/shaders_slang

This allows external SLANGP packs to work in Neo's portable folder without
requiring the original RetroArch installation layout.

Recommended layout:
  cHiDeScaler-Neo/
  └─ slangp/
     ├─ shaders_slang/
     └─ <any SLANGP preset pack>/

External shader packs are separate from Neo itself.
Please review the license and usage terms provided by each shader pack distributor.
