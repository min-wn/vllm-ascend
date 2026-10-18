# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.16)
project(gdn_q4_products LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(RUN_MODE "npu" CACHE STRING "Ascend C compilation mode")
set(SOC_VERSION "Ascend910B3" CACHE STRING "Target hardware")
set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}")
if(NOT ASCEND_CANN_PACKAGE_PATH)
    set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest")
endif()
include(${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake/ascendc.cmake)
include_directories(${CMAKE_BINARY_DIR})
ascendc_library(gdn_q4_products_kernels SHARED products_kernel.cpp)
target_include_directories(gdn_q4_products_kernels PRIVATE ${CMAKE_BINARY_DIR})

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(Torch REQUIRED)
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import pathlib, torch_npu; print(pathlib.Path(torch_npu.__file__).parent)"
    OUTPUT_VARIABLE TORCH_NPU_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY
)
add_library(gdn_q4_products MODULE torch_binding.cpp)
target_include_directories(gdn_q4_products PRIVATE
    ${TORCH_INCLUDE_DIRS} ${Python_INCLUDE_DIRS} ${TORCH_NPU_ROOT}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include
)
target_compile_options(gdn_q4_products PRIVATE ${TORCH_CXX_FLAGS})
target_link_directories(gdn_q4_products PRIVATE
    ${TORCH_INSTALL_PREFIX}/lib ${TORCH_NPU_ROOT}/lib ${ASCEND_CANN_PACKAGE_PATH}/lib64
)
target_link_libraries(gdn_q4_products PRIVATE
    gdn_q4_products_kernels ${TORCH_LIBRARIES} torch_python torch_npu ascendcl
)
target_link_options(gdn_q4_products PRIVATE
    "-Wl,-rpath,${TORCH_INSTALL_PREFIX}/lib:${TORCH_NPU_ROOT}/lib:${ASCEND_CANN_PACKAGE_PATH}/lib64:${CMAKE_BINARY_DIR}/lib"
)
set_target_properties(gdn_q4_products PROPERTIES
    PREFIX "" BUILD_RPATH "${TORCH_NPU_ROOT}/lib;${ASCEND_CANN_PACKAGE_PATH}/lib64;${CMAKE_BINARY_DIR}"
)
