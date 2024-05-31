#include <sycl/sycl.hpp>
#include "cinn_sycl_runtime_source.h"
typedef sycl::half float16;
#ifdef __cplusplus
extern "C" {
#endif
// CodeGenSYCL: NOTE: Auto-generated packed function
void fn_reshape_0_reshape_1_reshape_2_reshape_3_broadcast_to_5_scale_7_fill_constant_10_pow_11_subtract_6_broadcast_to_12_elementwise_mul_13_broadcast_to_14_elementwise_mul_15_broadcast_to_16_elementwise_add_17_elementwise_add_18_fill_constant_19_max_20_26_kernel(sycl::queue &Q, sycl::range<3> dimGrid, sycl::range<3> dimBlock, void** void_args) {
  const float*  batch_norm2d_52__b_0 = (float* )(*(void **)(void_args[0]));
  const float*  batch_norm2d_52__w_0 = (float* )(*(void **)(void_args[1]));
  const float*  batch_norm2d_52__w_1 = (float* )(*(void **)(void_args[2]));
  const float*  batch_norm2d_52__w_2 = (float* )(*(void **)(void_args[3]));
  const float*  conv2d_52__tmp_0 = (float* )(*(void **)(void_args[4]));
  const float*  relu_45__tmp_0 = (float* )(*(void **)(void_args[5]));
  float*  var_63 = (float* )(*(void **)(void_args[6]));
  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class space234_fn_reshape_0_reshape_1_reshape_2_reshape_3_broadcast_to_5_scale_7_fill_constant_10_pow_11_subtract_6_broadcast_to_12_elementwise_mul_13_broadcast_to_14_elementwise_mul_15_broadcast_to_16_elementwise_add_17_elementwise_add_18_fill_constant_19_max_20_26_kernel>(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item) [[intel::kernel_args_restrict]][[intel::max_work_group_size(1, 1, 1)]]
    {
      for (int32_t flat_i = 0; flat_i < 163840; flat_i += 1) {
        var_63[flat_i] = sycl::max(((batch_norm2d_52__w_0[(flat_i / 80)] * (cinn_sycl_rsqrt_fp32((batch_norm2d_52__w_2[((flat_i % 163840) / 80)] + 9.99999975e-06f)) * conv2d_52__tmp_0[flat_i])) + ((-1.00000000f * (batch_norm2d_52__w_0[(flat_i / 80)] * (cinn_sycl_rsqrt_fp32((batch_norm2d_52__w_2[((flat_i % 163840) / 80)] + 9.99999975e-06f)) * batch_norm2d_52__w_1[(flat_i / 80)]))) + (batch_norm2d_52__b_0[(flat_i / 80)] + relu_45__tmp_0[flat_i]))), 0.00000000f);
      };
    });
  });
}

#ifdef __cplusplus
}
#endif
