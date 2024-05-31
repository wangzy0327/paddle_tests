#include <sycl/sycl.hpp>
#include "cinn_sycl_runtime_source.h"
typedef sycl::half float16;
#ifdef __cplusplus
extern "C" {
#endif
// CodeGenSYCL: NOTE: Auto-generated packed function
void fn_broadcast_to_24_elementwise_add_25_30_kernel(sycl::queue &Q, sycl::range<3> dimGrid, sycl::range<3> dimBlock, void** void_args) {
  const float*  linear_1__b_0 = (float* )(*(void **)(void_args[0]));
  const float*  var_71 = (float* )(*(void **)(void_args[1]));
  float*  var_73 = (float* )(*(void **)(void_args[2]));
  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class space233_fn_broadcast_to_24_elementwise_add_25_30_kernel>(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item) [[intel::kernel_args_restrict]][[intel::max_work_group_size(1, 1, 1)]]
    {
      for (int32_t flat_i = 0; flat_i < 128; flat_i += 1) {
        var_73[flat_i] = (var_71[flat_i] + linear_1__b_0[flat_i]);
      };
    });
  });
}

#ifdef __cplusplus
}
#endif
