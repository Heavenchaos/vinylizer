cmake_minimum_required(VERSION 3.22)
project(vinylizer-cpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
add_compile_definitions(_USE_MATH_DEFINES)

# ── GPU architecture ──────────────────────────────────────────
# 发布版面向玩家，必须覆盖主流 NVIDIA 显卡，否则玩家机器上会直接报
# "no kernel image is available for execution on the device"。
# CUDA 13.3 支持的最低架构是 sm_75（sm_52 已被移除），实际集合：
#   75  = RTX 20 系 / GTX 16 系（Turing）
#   86  = RTX 30 系（Ampere 消费级）；专业卡 RTX A4000/A5000/A6000 同为 86
#   89  = RTX 40 系（Ada）
#   120 = RTX 50 系（Blackwell 消费级）
# 每档用 `-real` 生成原生 cubin（性能最好）；末尾 `120-virtual` 额外保留 PTX，
# 供未来新架构（如 sm_121+）由驱动 JIT 前向兼容。
# 不编 sm_80/sm_90：A100/H100 等数据中心卡不是玩家硬件，编入只增体积与构建时间。
# 构建机为 RTX 5090（sm_120），运行时会选中 120 的原生 cubin，行为与单架构构建一致。
# 注：det1（deterministic-reduce=1）下归约顺序与架构无关（sm_52 ≡ sm_86 曾验证 bit-exact），
#     sm 架构不影响数值结果。
set(CMAKE_CUDA_ARCHITECTURES "75-real;86-real;89-real;120-real;120-virtual" CACHE STRING "CUDA target architectures")

# ── Compiler optimization flags ───────────────────────────────
if(MSVC)
    # Force UTF-8 source encoding (避免中文注释被 GBK 误解析导致编译错误)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler> $<$<COMPILE_LANGUAGE:CUDA>:/utf-8>)
    # CPU: fast floating-point, AVX2 (C/C++ only, not CUDA)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/O2>)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/fp:fast>)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/arch:AVX2>)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/DNDEBUG>)
    # CUDA: fast math (verified safe with sm_52, see AGENTS.md)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
elseif(UNIX)
    # Linux/云上部署（AutoDL 等）：与 MSVC 分支等价的优化语义（MSVC 分支保持一字不动，Windows baseline 不受影响）
    # gcc 默认 UTF-8，无需 /utf-8；-DNDEBUG 与 /DNDEBUG 等价
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-O3>)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-ffast-math>)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-mavx2>)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-DNDEBUG>)
    # CUDA: fast math（与 MSVC 分支一致）
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
endif()

# ── Dependencies ──────────────────────────────────────────────
# 发布物是 GUI（vinylizer.exe）；vinylizer-cli.exe 是无 Qt 的命令行版本（回归/批处理）。
option(VINYLIZER_ENABLE_GUI "Build the Qt6 GUI (vinylizer.exe)" ON)
option(VINYLIZER_NODEBUG "Disable all debug logging and statistics for max performance" OFF)
if(VINYLIZER_NODEBUG)
    add_compile_definitions(__NODEBUG__)
    message(STATUS "Vinylizer: __NODEBUG__ enabled - all logging disabled (except VIN_ERROR)")
endif()

# OpenCV (via vcpkg)
find_package(OpenCV REQUIRED)

# CUDA
find_package(CUDAToolkit REQUIRED)

# Qt6 (for GUI)
if(VINYLIZER_ENABLE_GUI)
    set(CMAKE_AUTOMOC ON)
    find_package(Qt6 REQUIRED COMPONENTS Widgets OpenGL OpenGLWidgets Concurrent)
endif()

# ── Source files ──────────────────────────────────────────────
set(COMMON_SOURCES
    src/common/canvas.cpp
    src/common/json_to_image.cpp
)

set(PREPROCESS_SOURCES
    src/preprocess/preprocessor.cpp
)

set(CORE_SOURCES
    src/core/optimizer.cpp
    src/core/pipeline.cpp
    src/core/initializer/variance_initializer.cpp
    src/core/initializer/kmedoids_initializer.cpp
)

set(CUDA_SOURCES
    src/cuda/render_kernel.cu
    src/cuda/canvas_render.cu
    src/cuda/template_sampling.cu
    src/cuda/kmedoids_kernels.cu
    src/core/diff_renderer.cu
)

set(OUTPUT_SOURCES
    src/output/json_writer.cpp
)

set(INJECT_SOURCES
    src/inject/inject.cpp
)

set(SHARED_SOURCES
    ${COMMON_SOURCES}
    ${PREPROCESS_SOURCES}
    ${CORE_SOURCES}
    ${CUDA_SOURCES}
    ${OUTPUT_SOURCES}
)
# inject 模块依赖 Windows API（FH6 内存注入），仅 Windows 编译；Linux/云上不包含
if(MSVC)
    list(APPEND SHARED_SOURCES ${INJECT_SOURCES})
endif()

# ── CLI executable (no Qt) ───────────────────────────────────
add_executable(vinylizer-cli
    src/main.cpp
    ${SHARED_SOURCES}
)

set_target_properties(vinylizer-cli PROPERTIES
    WIN32_EXECUTABLE FALSE
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(vinylizer-cli PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/vendor
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(vinylizer-cli PRIVATE
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    opencv_objdetect
    CUDA::cudart
)
if(UNIX)
    # Linux/云上：--preview 的 imshow/namedWindow 需要 HighGUI（Windows 由 vcpkg 隐式提供）
    target_link_libraries(vinylizer-cli PRIVATE opencv_highgui)
endif()

# ── GUI executable (release artifact) ────────────────────────
if(VINYLIZER_ENABLE_GUI)
    set(GUI_SOURCES
        src/gui/main_gui.cpp
        src/gui/main_window.cpp
        src/gui/main_window.h
        src/gui/optimize_page.cpp
        src/gui/optimize_page.h
        src/gui/inject_page.cpp
        src/gui/inject_page.h
        src/gui/preview_gl_widget.cpp
        src/gui/preview_gl_widget.h
        src/gui/optimize_worker.cpp
        src/gui/optimize_worker.h
        src/gui/inject_worker.cpp
        src/gui/inject_worker.h
        src/common/preview_render.cpp
    )

    add_executable(vinylizer
        ${GUI_SOURCES}
        ${SHARED_SOURCES}
    )

    set_target_properties(vinylizer PROPERTIES
        WIN32_EXECUTABLE TRUE
        CUDA_SEPARABLE_COMPILATION ON
    )

    target_include_directories(vinylizer PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/vendor
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    target_link_libraries(vinylizer PRIVATE
        opencv_core
        opencv_imgproc
        opencv_imgcodecs
        opencv_objdetect
        CUDA::cudart
        Qt6::Widgets
        Qt6::OpenGL
        Qt6::OpenGLWidgets
        Qt6::Concurrent
    )

    target_compile_definitions(vinylizer PRIVATE VINYLIZER_GUI=1)
endif()

# ── Windows dist packaging ────────────────────────────────────
# 产出 release/dist/：玩家解压即用（不需要设置任何 PATH）。
#   cmake --build build --target dist --config Release
# 内容：vinylizer.exe + vinylizer-cli.exe + Qt6 运行库（含 platforms 插件）
#       + OpenCV 运行库（opencv_world4120.dll）
#       + MSVC 运行库（app-local：从 VC/Redist/MSVC/*/x64/Microsoft.VC143.CRT 拷）
# CUDA 运行库是静态链接的（CUDA_SEPARABLE_COMPILATION），无需单独拷贝；
# 实际依赖以 `dumpbin /dependents dist\vinylizer*.exe` 为准。
if(WIN32 AND VINYLIZER_ENABLE_GUI)
    set(VINYLIZER_DIST_DIR "${CMAKE_SOURCE_DIR}/dist")

    # 从编译器路径推导 VC/Redist/MSVC/<ver>/x64/Microsoft.VC143.CRT（app-local CRT）
    set(_vctools "${CMAKE_CXX_COMPILER}")
    foreach(_i RANGE 4)
        get_filename_component(_vctools "${_vctools}" DIRECTORY)
    endforeach()
    file(GLOB _msvc_crt_dirs "${_vctools}/../../Redist/MSVC/*/x64/Microsoft.VC143.CRT")
    list(SORT _msvc_crt_dirs)
    list(REVERSE _msvc_crt_dirs)
    list(GET _msvc_crt_dirs 0 _msvc_crt_dir)

    add_custom_target(dist
        COMMAND ${CMAKE_COMMAND} -E rm -rf "${VINYLIZER_DIST_DIR}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${VINYLIZER_DIST_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:vinylizer-cli>" "${VINYLIZER_DIST_DIR}/"
        COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:vinylizer>" "${VINYLIZER_DIST_DIR}/"
        COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:opencv_core>" "${VINYLIZER_DIST_DIR}/"
        COMMAND "${Qt6_DIR}/../../../bin/windeployqt.exe"
                --no-translations --no-opengl-sw
                "${VINYLIZER_DIST_DIR}/vinylizer.exe"
        # offscreen 平台插件：headless smoke test（vinylizer.exe -platform offscreen）
        COMMAND ${CMAKE_COMMAND} -E copy
                "${Qt6_DIR}/../../../plugins/platforms/qoffscreen.dll"
                "${VINYLIZER_DIST_DIR}/platforms/"
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${_msvc_crt_dir}" "${VINYLIZER_DIST_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy
                "${CMAKE_SOURCE_DIR}/README.md" "${VINYLIZER_DIST_DIR}/README.md"
        COMMENT "Packaging dist/ (exes + Qt + OpenCV + MSVC CRT)"
        VERBATIM
        DEPENDS vinylizer vinylizer-cli
    )
endif()

# ── Test executable ───────────────────────────────────────────
# Removed — test_verify.cpp deleted during code simplification
