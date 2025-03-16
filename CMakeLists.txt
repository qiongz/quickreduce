cmake_minimum_required(VERSION 3.25)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/find)

project(quickreduce LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Detect if the project is being build within a project or standalone.
if(PROJECT_IS_TOP_LEVEL)
    # Configure the build path
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

    # Nice hack to automatically ignore the build directory
    file(WRITE ${CMAKE_BINARY_DIR}/.gitignore "*")
endif()

# CMake Config for packages
if(POLICY CMP0135)
	cmake_policy(SET CMP0135 NEW)
	set(CMAKE_POLICY_DEFAULT_CMP0135 NEW)
endif()

# =============================================================
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
find_package(MPI REQUIRED)

# REF: https://rocm.docs.amd.com/en/latest/understand/cmake_packages.html
find_package(hip REQUIRED)
find_package(RCCL REQUIRED)

# Resolve architectures
set(CMAKE_HIP_ARCHITECTURES "gfx942")
foreach(ARCH IN LISTS CMAKE_HIP_ARCHITECTURES)
  set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -D__${ARCH}__")
endforeach()


# =============================================================
# SOURCE
add_library(quickreduce INTERFACE)
target_include_directories(quickreduce SYSTEM INTERFACE csrc)
target_link_libraries(quickreduce INTERFACE hip::device)


# =============================================================
# TEST
add_custom_target(build_tests)

function(build_test name)
    add_executable(${name} test/${name}.hip)
    target_include_directories(${name} PUBLIC ${RCCL_INCLUDE_DIRS})
    target_link_libraries(${name} quickreduce hip::device MPI::MPI_CXX ${RCCL_LIBRARIES})
    target_compile_options(${name} PUBLIC -Wno-unused-result)
    add_dependencies(build_tests ${name})
endfunction()

build_test(oneshot_test)
build_test(twoshot_test)
build_test(twoshot_fp8_test)
build_test(twoshot_q4_test)
build_test(twoshot_q8_test)
build_test(twoshot_q6_test)
