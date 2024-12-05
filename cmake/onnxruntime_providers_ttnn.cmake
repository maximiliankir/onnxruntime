message("Building the Tenstorrent TTNN provider")

add_compile_definitions(USE_TTNN=1)

file(GLOB_RECURSE
  onnxruntime_providers_ttnn_cc_srcs CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/core/providers/ttnn/*.h"
  "${ONNXRUNTIME_ROOT}/core/providers/ttnn/*.cc"
)

source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_ttnn_cc_srcs})
onnxruntime_add_static_library(onnxruntime_providers_ttnn ${onnxruntime_providers_ttnn_cc_srcs})
onnxruntime_add_include_to_target(onnxruntime_providers_ttnn onnxruntime_common onnxruntime_framework onnx onnx_proto protobuf::libprotobuf-lite flatbuffers::flatbuffers Boost::mp11)
target_link_libraries(onnxruntime_providers_ttnn)

# Add Dependencies and Properties for the compilation
# TODO

install(TARGETS onnxruntime_providers_ttnn
        ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
        FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
