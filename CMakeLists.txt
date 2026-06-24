cmake_minimum_required(VERSION 3.22)
project(vinylizer-cpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
add_compile_definitions(_USE_MATH_DEFINES)

# ── GPU architecture ──────────────────────────────────────────
set(CMAKE_CUDA_ARCHITECTURES 86 89 CACHE STRING "CUDA target architectures")

# ── Compiler optimization flags ───────────────────────────────
if(MSVC)
    # CPU: fast floating-point, AVX2 (C/C++ only, not CUDA)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/O2>)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/fp:fast>)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/arch:AVX2>)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/DNDEBUG>)
    # CUDA: fast math
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
endif()

# ── Dependencies ──────────────────────────────────────────────
option(VINYLIZER_ENABLE_GUI "Enable Qt6 GUI" OFF)

# OpenCV (via vcpkg)
find_package(OpenCV REQUIRED)

# CUDA
find_package(CUDAToolkit REQUIRED)

# Qt6 (for GUI)
if(VINYLIZER_ENABLE_GUI)
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTORCC ON)
    set(CMAKE_AUTOUIC ON)
    find_package(Qt6 REQUIRED COMPONENTS Widgets OpenGL OpenGLWidgets Concurrent)
endif()

# ── Source files ──────────────────────────────────────────────
set(COMMON_SOURCES
    src/common/canvas.cpp
)

set(PREPROCESS_SOURCES
    src/preprocess/preprocessor.cpp
)

set(CORE_SOURCES
    src/core/optimizer.cpp
    src/core/pipeline.cpp
)

set(CUDA_SOURCES
    src/cuda/render_kernel.cu
    src/cuda/canvas_render.cu
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
    ${INJECT_SOURCES}
)

# ── Main executable ──────────────────────────────────────────
if(VINYLIZER_ENABLE_GUI)
    set(GUI_SOURCES
        src/gui/main_gui.cpp
        src/gui/render_canvas.cpp
        src/gui/optimize_worker.cpp
        src/gui/inject_worker.cpp
        icon/resources.qrc
    )

    add_executable(vinylizer
        ${GUI_SOURCES}
        ${SHARED_SOURCES}
        icon/app.rc
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

    # Windows: auto-deploy Qt DLLs
    add_custom_command(TARGET vinylizer POST_BUILD
        COMMAND "${Qt6_DIR}/../../../bin/windeployqt.exe"
            $<TARGET_FILE:vinylizer>
            --no-translations
            --no-opengl-sw
    )
else()
    add_executable(vinylizer
        src/main.cpp
        ${SHARED_SOURCES}
    )

    set_target_properties(vinylizer PROPERTIES
        WIN32_EXECUTABLE FALSE
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
    )
endif()

# ── Test executable ───────────────────────────────────────────
# Removed — test_verify.cpp deleted during code simplification
