C headers of whisper.cpp **v1.9.4** (commit 927cfce34f31707e17f2bff35c349632fb9e2c3a), the same commit as the pinned
base image: `parakeet.h`, `whisper.h` and `ggml/*.h` from https://github.com/ggml-org/whisper.cpp (MIT licence, see
`LICENSE.whisper.cpp`). They are not compiled into the server; the ctypes bindings in `wyoming_vulkan/engines/` mirror
their structs, and `tests/check_abi.sh` compiles a layout check against them.
