#include <sycl/sycl.hpp>
#include "cinn_sycl_runtime_source.h"
typedef sycl::half float16;
#ifdef __cplusplus
extern "C" {
#endif
// CodeGenSYCL: NOTE: Auto-generated packed function
void fn_reshape_0_reshape_1_reshape_2_reshape_3_reshape_4_reshape_5_reshape_6_reshape_7_broadcast_to_10_scale_12_fill_constant_20_broadcast_to_13_scale_15_fill_constant_22_pow_21_pow_23_subtract_11_broadcast_to_24_subtract_14_broadcast_to_26_elementwise_mul_25_broadcast_to_28_elementwise_mul_27_broadcast_to_30_elementwise_mul_29_broadcast_to_32_elementwise_mul_31_broadcast_to_34_elementwise_add_35_elementwise_add_33_fill_constant_37_elementwise_add_36_max_38_40_kernel(sycl::queue &Q, sycl::range<3> dimGrid, sycl::range<3> dimBlock, void** void_args) {
  const float*  batch_norm2d_11__b_0 = (float* )(*(void **)(void_args[0]));
  const float*  batch_norm2d_11__w_0 = (float* )(*(void **)(void_args[1]));
  const float*  batch_norm2d_11__w_1 = (float* )(*(void **)(void_args[2]));
  const float*  batch_norm2d_11__w_2 = (float* )(*(void **)(void_args[3]));
  const float*  batch_norm2d_14__b_0 = (float* )(*(void **)(void_args[4]));
  const float*  batch_norm2d_14__w_0 = (float* )(*(void **)(void_args[5]));
  const float*  batch_norm2d_14__w_1 = (float* )(*(void **)(void_args[6]));
  const float*  batch_norm2d_14__w_2 = (float* )(*(void **)(void_args[7]));
  const float*  conv2d_14__tmp_0 = (float* )(*(void **)(void_args[8]));
  const float*  conv2d_13__tmp_0 = (float* )(*(void **)(void_args[9]));
  float*  var_107 = (float* )(*(void **)(void_args[10]));
  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class space159_fn_reshape_0_reshape_1_reshape_2_reshape_3_reshape_4_reshape_5_reshape_6_reshape_7_broadcast_to_10_scale_12_fill_constant_20_broadcast_to_13_scale_15_fill_constant_22_pow_21_pow_23_subtract_11_broadcast_to_24_subtract_14_broadcast_to_26_elementwise_mul_25_broadcast_to_28_elementwise_mul_27_broadcast_to_30_elementwise_mul_29_broadcast_to_32_elementwise_mul_31_broadcast_to_34_elementwise_add_35_elementwise_add_33_fill_constant_37_elementwise_add_36_max_38_40_kernel>(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item) [[intel::kernel_args_restrict]][[intel::max_work_group_size(1, 1, 1)]]
    {
      for (int32_t flat_i = 0; flat_i < 614400; flat_i += 1) {
        var_107[flat_i] = sycl::max(((batch_norm2d_14__w_0[(flat_i / 1200)] * (cinn_sycl_rsqrt_fp32((batch_norm2d_14__w_2[((flat_i % 614400) / 1200)] + 9.99999975e-06f)) * conv2d_13__tmp_0[flat_i])) + ((-1.00000000f * (batch_norm2d_14__w_0[(flat_i / 1200)] * (cinn_sycl_rsqrt_fp32((batch_norm2d_14__w_2[((flat_i % 614400) / 1200)] + 9.99999975e-06f)) * batch_norm2d_14__w_1[(flat_i / 1200)]))) + (batch_norm2d_14__b_0[(flat_i / 1200)] + ((batch_norm2d_11__w_0[(flat_i / 1200)] * (cinn_sycl_rsqrt_fp32((batch_norm2d_11__w_2[((flat_i % 614400) / 1200)] + 9.99999975e-06f)) * conv2d_14__tmp_0[flat_i])) + ((-1.00000000f * (batch_norm2d_11__w_0[(flat_i / 1200)] * (cinn_sycl_rsqrt_fp32((batch_norm2d_11__w_2[((flat_i % 614400) / 1200)] + 9.99999975e-06f)) * batch_norm2d_11__w_1[(flat_i / 1200)]))) + batch_norm2d_11__b_0[(flat_i / 1200)]))))), 0.00000000f);
      };
    });
  });
}

#ifdef __cplusplus
}
#endif
