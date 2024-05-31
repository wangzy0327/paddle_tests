#include <sycl/sycl.hpp>
#include "cinn_sycl_runtime_source.h"
typedef sycl::half float16;
#ifdef __cplusplus
extern "C" {
#endif
// CodeGenSYCL: NOTE: Auto-generated packed function
void fn_broadcast_to_3_elementwise_add_4_7_kernel(sycl::queue &Q, sycl::range<3> dimGrid, sycl::range<3> dimBlock, void** void_args) {
  const float*  fc_offset = (float* )(*(void **)(void_args[0]));
  const float*  var_15 = (float* )(*(void **)(void_args[1]));
  float*  var_17 = (float* )(*(void **)(void_args[2]));
  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class space131_fn_broadcast_to_3_elementwise_add_4_7_kernel>(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item) [[intel::kernel_args_restrict]][[intel::max_work_group_size(1, 1, 1)]]
    {
      for (int32_t flat_i = 0; flat_i < 1000; flat_i += 1) {
        var_17[flat_i] = (var_15[flat_i] + fc_offset[flat_i]);
      };
    });
  });
}

#ifdef __cplusplus
}
#endif
