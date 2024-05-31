#include <sycl/sycl.hpp>
#include "cinn_sycl_runtime_source.h"
typedef sycl::half float16;
#ifdef __cplusplus
extern "C" {
#endif
// CodeGenSYCL: NOTE: Auto-generated packed function
void fn_fill_constant_8_fill_constant_16_pow_17_38_kernel(sycl::queue &Q, sycl::range<3> dimGrid, sycl::range<3> dimBlock, void** void_args) {
  float*  var_89 = (float* )(*(void **)(void_args[0]));
  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class space138_fn_fill_constant_8_fill_constant_16_pow_17_38_kernel>(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item) [[intel::kernel_args_restrict]][[intel::max_work_group_size(1, 1, 1)]]
    {
      for (int32_t flat_i = 0; flat_i < 256; flat_i += 1) {
        var_89[flat_i] = cinn_sycl_rsqrt_fp32(9.99999975e-06f);
      };
    });
  });
}

#ifdef __cplusplus
}
#endif
