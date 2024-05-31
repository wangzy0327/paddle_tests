#include <sycl/sycl.hpp>
#include "cinn_sycl_runtime_source.h"
typedef sycl::half float16;
#ifdef __cplusplus
extern "C" {
#endif
// CodeGenSYCL: NOTE: Auto-generated packed function
void fn_fill_constant_1_greater_equal_2_cast_3_4_kernel(sycl::queue &Q, sycl::range<3> dimGrid, sycl::range<3> dimBlock, void** void_args) {
  const float*  var_1 = (float* )(*(void **)(void_args[0]));
  uint8_t*  var_7 = (uint8_t* )(*(void **)(void_args[1]));
  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class space130_fn_fill_constant_1_greater_equal_2_cast_3_4_kernel>(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item) [[intel::kernel_args_restrict]][[intel::max_work_group_size(1, 1, 1)]]
    {
      for (int32_t flat_i = 0; flat_i < 1280; flat_i += 1) {
        var_7[flat_i] = ((uint8_t)((var_1[flat_i] >= 0.200000003f)));
      };
    });
  });
}

#ifdef __cplusplus
}
#endif
