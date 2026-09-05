Place an authorized DLSS Neural Rendering runtime here as:

  nvngx_dlssnr.dll

The standard Neo package already contains backend.json and
neo_dlssnr_backend.dll. No separate Bridge Pack is required. Keep the runtime
DLL filename unchanged, place it in this folder, and restart Neo.

The Neo host never loads this DLL directly. It is loaded privately by the
hash-pinned neo_dlssnr_backend.dll bridge only after an explicit probe/render
enable request.

Alternative user/mod locations understood by the current bridge are:

  runtime\community\nvngx_dlssnr.dll
  runtime\mod\nvngx_dlssnr.dll

Do not copy or replace NGX DLLs in Windows, a game directory, or another
application. Runtime binaries are not part of the Neo source license and are
not redistributed in this source package.
