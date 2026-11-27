if (NOT GGML_SYSTEM_ARCH STREQUAL "x86" OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "GGML_XDNA requires x86-64")
endif()
if (WIN32 AND NOT MSVC)
    message(FATAL_ERROR "GGML_XDNA requires a compiler with the MSVC ABI on Windows")
endif()
if (NOT WIN32 AND NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "GGML_XDNA requires Windows or Linux")
endif()

set(GGML_XDNA_XRT_ROOT "" CACHE PATH "Path to the XRT SDK")
find_path(GGML_XDNA_XRT_INCLUDE_DIR xrt/xrt_device.h
    HINTS "${GGML_XDNA_XRT_ROOT}" "$ENV{XILINX_XRT}"
    PATHS /opt/xilinx/xrt
    PATH_SUFFIXES include include/xrt xrt)
find_library(GGML_XDNA_XRT_LIBRARY xrt_coreutil
    HINTS "${GGML_XDNA_XRT_ROOT}" "$ENV{XILINX_XRT}"
    PATHS /opt/xilinx/xrt
    PATH_SUFFIXES lib lib64)

if (NOT GGML_XDNA_XRT_INCLUDE_DIR OR NOT GGML_XDNA_XRT_LIBRARY)
    message(FATAL_ERROR "Install the XRT development package or set GGML_XDNA_XRT_ROOT to the XRT SDK")
endif()

ggml_add_backend_library(ggml-xdna ggml-xdna.cpp)
target_include_directories(ggml-xdna SYSTEM PRIVATE "${GGML_XDNA_XRT_INCLUDE_DIR}")
target_link_libraries(ggml-xdna PRIVATE "${GGML_XDNA_XRT_LIBRARY}")
target_compile_features(ggml-xdna PRIVATE cxx_std_17)
if (MSVC)
    # XRT uses __cplusplus to select the C++17 ABI.
    target_compile_options(ggml-xdna PRIVATE /Zc:__cplusplus /EHsc)
endif()
