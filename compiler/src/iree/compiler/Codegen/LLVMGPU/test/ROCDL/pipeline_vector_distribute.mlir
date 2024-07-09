// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 \
// RUN:   --iree-codegen-llvmgpu-use-vector-distribution --iree-llvmgpu-enable-prefetch=true \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-llvmgpu-select-lowering-strategy, func.func(iree-llvmgpu-lower-executable-target)))))" \
// RUN:   %s | FileCheck %s

// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx1100 \
// RUN:   --iree-codegen-llvmgpu-use-vector-distribution --iree-llvmgpu-enable-prefetch=true \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-llvmgpu-select-lowering-strategy, func.func(iree-llvmgpu-lower-executable-target)))))" \
// RUN:   %s | FileCheck %s --check-prefix=RDNA3

// TODO: This test is still using the legacy LLVMGPU kernel config. This needs
// to be migrated to the rocdl heuristics, but for now is just physically
// located here.

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @matmul_256x256x256_f16_f32 {
hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb">) {
  hal.executable.export @matmul_256x256x256_f16_f32 layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @matmul_256x256x256_f16_f32() {
      %cst = arith.constant 0.000000e+00 : f32
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256x256xf16>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256x256xf16>>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<256x256xf32>>
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x256xf16>> -> tensor<256x256xf16>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x256xf16>> -> tensor<256x256xf16>
      %5 = tensor.empty() : tensor<256x256xf32>
      %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<256x256xf32>) -> tensor<256x256xf32>
      %7 = linalg.matmul ins(%3, %4 : tensor<256x256xf16>, tensor<256x256xf16>) outs(%6 : tensor<256x256xf32>) -> tensor<256x256xf32>
      flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : tensor<256x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<256x256xf32>>
      return
    }
  }
}
}

// Basic pipeline test to make sure it generates the instructions we expect.

//       CHECK: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUVectorDistribute workgroup_size = [128, 2, 1] subgroup_size = 64
//  CHECK-SAME:   mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>,
//  CHECK-SAME:     subgroup_m_count = 2, subgroup_n_count = 2>
//  CHECK-SAME:   prefetch_shared_memory

//    CHECK-LABEL: func.func @matmul_256x256x256_f16_f32()
//     CHECK-SAME:    translation_info = #[[$TRANSLATION]]
//          CHECK:   scf.for {{.*}} = %c0 to %c256 step %c128 iter_args({{.*}}) -> (vector<2x2x1x1x4x1xf32>)
// Each subgroup handles 2 * 2 tiles, and for each tile we accumulate 8 times
// along the K dimension. So in total 32 mfma ops.
// CHECK-COUNT-32:     amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
//          CHECK:     scf.yield %{{.+}} : vector<2x2x1x1x4x1xf32>
//  CHECK-COUNT-4:   vector.transfer_write {{.+}} {in_bounds = [true, true]} : vector<4x1xf32>, memref<256x256xf32, #hal.descriptor_type<storage_buffer>>

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @matmul_256x256x256_f16_f16 {
hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb">) {
  hal.executable.export @matmul_256x256x256_f16_f16 layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @matmul_256x256x256_f16_f16() {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256x256xf16>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256x256xf16>>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<256x256xf16>>
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x256xf16>> -> tensor<256x256xf16>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x256xf16>> -> tensor<256x256xf16>
      %5 = tensor.empty() : tensor<256x256xf16>
      %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<256x256xf16>) -> tensor<256x256xf16>
      %7 = linalg.matmul ins(%3, %4 : tensor<256x256xf16>, tensor<256x256xf16>) outs(%6 : tensor<256x256xf16>) -> tensor<256x256xf16>
      flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : tensor<256x256xf16> -> !flow.dispatch.tensor<writeonly:tensor<256x256xf16>>
      return
    }
  }
}
}

//       CHECK: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUVectorDistribute workgroup_size = [128, 2, 1] subgroup_size = 64
//  CHECK-SAME:   mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>,
//  CHECK-SAME:     subgroup_m_count = 2, subgroup_n_count = 2>
//  CHECK-SAME:   prefetch_shared_memory

//    CHECK-LABEL: func.func @matmul_256x256x256_f16_f16()
//     CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//          CHECK:   scf.for {{.*}} = %c0 to %c256 step %c128 iter_args(%[[ARG:.+]] = {{.*}}) -> (vector<2x2x1x1x4x1xf16>)
//          CHECK:     arith.extf %[[ARG]] {{.*}} : vector<2x2x1x1x4x1xf16> to vector<2x2x1x1x4x1xf32>
// CHECK-COUNT-32:     amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
//          CHECK:     %[[TRUNC:.+]] = arith.truncf %{{.*}} : vector<2x2x1x1x4x1xf32> to vector<2x2x1x1x4x1xf16>
//          CHECK:     scf.yield %[[TRUNC]] : vector<2x2x1x1x4x1xf16>
//  CHECK-COUNT-4:   vector.transfer_write {{.+}} {in_bounds = [true, true]} : vector<4x1xf16>, memref<256x256xf16, #hal.descriptor_type<storage_buffer>>

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @expanded_matmul_transpose_b_executable {
hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb">) {
  hal.executable.export @expanded_matmul_transpose_b layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @expanded_matmul_transpose_b() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
          : !flow.dispatch.tensor<readonly:tensor<2x64x2048xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
          : !flow.dispatch.tensor<readonly:tensor<10x64x2048xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0)
          : !flow.dispatch.tensor<writeonly:tensor<2x10x64x64xf16>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [2, 64, 2048], strides = [1, 1, 1]
          : !flow.dispatch.tensor<readonly:tensor<2x64x2048xf16>> -> tensor<2x64x2048xf16>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [10, 64, 2048], strides = [1, 1, 1]
          : !flow.dispatch.tensor<readonly:tensor<10x64x2048xf16>> -> tensor<10x64x2048xf16>

        %5 = tensor.empty() : tensor<2x10x64x64xf16>
        %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<2x10x64x64xf16>) -> tensor<2x10x64x64xf16>
        %7 = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>,
            affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>,
            affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
        } ins(%3, %4 : tensor<2x64x2048xf16>, tensor<10x64x2048xf16>) outs(%6 : tensor<2x10x64x64xf16>) {
        ^bb0(%lhs: f16, %rhs: f16, %out: f16):
          %mul = arith.mulf %lhs, %rhs : f16
          %add = arith.addf %mul, %out : f16
          linalg.yield %add : f16
        } -> tensor<2x10x64x64xf16>

        flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 10, 64, 64], strides = [1, 1, 1, 1]
          : tensor<2x10x64x64xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x10x64x64xf16>>
        return
      }
    }
  }
}

//          CHECK: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 64
//     CHECK-SAME: prefetch_shared_memory

//          CHECK: func @expanded_matmul_transpose_b
//     CHECK-SAME:     translation_info = #[[TRANSLATION]]
// This has more than 2 iteartions. So we have prefetching enabled for this case. Due to
// prefetching, we have one iteration peeled of so upper bound is 2048 - 128 = 1920.
//          CHECK:   scf.for {{.*}} = %c0 to %c15 step %c1 iter_args(%[[ARG:.+]] = {{.*}}) -> (vector<4x1x1x1x4x1xf16>)
//          CHECK:     arith.extf %[[ARG]] {{.*}} : vector<4x1x1x1x4x1xf16> to vector<4x1x1x1x4x1xf32>
// CHECK-COUNT-32:     amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
//          CHECK:     %[[TRUNC:.+]] = arith.truncf %{{.*}} : vector<4x1x1x1x4x1xf32> to vector<4x1x1x1x4x1xf16>
//          CHECK:     scf.yield %[[TRUNC]] : vector<4x1x1x1x4x1xf16>
// CHECK-COUNT-32:   amdgpu.mfma
//  CHECK-COUNT-4:   vector.transfer_write {{.+}} {in_bounds = [true, true]} : vector<4x1xf16>, memref<2x10x64x64xf16, #hal.descriptor_type<storage_buffer>>

// -----

// Basic i8, i8 -> i32 matmul.

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @matmul_256x256x256_i8_i32 {
hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb">) {
  hal.executable.export @matmul_256x256x256_i8_i32 layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @matmul_256x256x256_i8_i32() {
      %cst = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256x256xi8>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256x256xi8>>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<256x256xi32>>
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x256xi8>> -> tensor<256x256xi8>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x256xi8>> -> tensor<256x256xi8>
      %5 = tensor.empty() : tensor<256x256xi32>
      %6 = linalg.fill ins(%cst : i32) outs(%5 : tensor<256x256xi32>) -> tensor<256x256xi32>
      %7 = linalg.matmul ins(%3, %4 : tensor<256x256xi8>, tensor<256x256xi8>) outs(%6 : tensor<256x256xi32>) -> tensor<256x256xi32>
      flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : tensor<256x256xi32> -> !flow.dispatch.tensor<writeonly:tensor<256x256xi32>>
      return
    }
  }
}
}

// Make sure it generates the mfma instructions we expect for integer inputs.

//       CHECK: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUVectorDistribute workgroup_size = [128, 2, 1] subgroup_size = 64
//  CHECK-SAME:   mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mma_layout<MFMA_I8_16x16x32_I32>,
//  CHECK-SAME:     subgroup_m_count = 2, subgroup_n_count = 2>
//  CHECK-SAME:   prefetch_shared_memory

//    CHECK-LABEL: func.func @matmul_256x256x256_i8_i32()
//     CHECK-SAME:    translation_info = #[[$TRANSLATION]]
// Each subgroup handles 2 * 2 tiles, and for each tile we accumulate 8 times
// along the K dimension. So in total 32 mfma ops.
// CHECK-COUNT-32:     amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
//  CHECK-COUNT-4:   vector.transfer_write {{.+}} {in_bounds = [true, true]} : vector<4x1xi32>, memref<256x256xi32, #hal.descriptor_type<storage_buffer>>

// -----

// Basic i8, i8 -> i32 matmul_transpose_b.

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @matmul_transpose_b_256x256x256_i8_i32 {
hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb">) {
  hal.executable.export @matmul_transpose_b_256x256x256_i8_i32 layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @matmul_transpose_b_256x256x256_i8_i32() {
      %cst = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256x256xi8>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256x256xi8>>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<256x256xi32>>
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x256xi8>> -> tensor<256x256xi8>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x256xi8>> -> tensor<256x256xi8>
      %5 = tensor.empty() : tensor<256x256xi32>
      %6 = linalg.fill ins(%cst : i32) outs(%5 : tensor<256x256xi32>) -> tensor<256x256xi32>
      %7 = linalg.matmul_transpose_b ins(%3, %4 : tensor<256x256xi8>, tensor<256x256xi8>) outs(%6 : tensor<256x256xi32>) -> tensor<256x256xi32>
      flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : tensor<256x256xi32> -> !flow.dispatch.tensor<writeonly:tensor<256x256xi32>>
      return
    }
  }
}
}

// Make sure it generates the mfma instructions we expect for integer inputs.

//       CHECK: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUVectorDistribute workgroup_size = [128, 2, 1] subgroup_size = 64
//  CHECK-SAME:   mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mma_layout<MFMA_I8_16x16x32_I32>,
//  CHECK-SAME:     subgroup_m_count = 2, subgroup_n_count = 2>
//  CHECK-SAME:   prefetch_shared_memory

//    CHECK-LABEL: func.func @matmul_transpose_b_256x256x256_i8_i32()
//     CHECK-SAME:    translation_info = #[[$TRANSLATION]]
// Each subgroup handles 2 * 2 tiles, and for each tile we accumulate 8 times
// along the K dimension. So in total 32 mfma ops.
// CHECK-COUNT-32:     amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
//  CHECK-COUNT-4:   vector.transfer_write {{.+}} {in_bounds = [true, true]} : vector<4x1xi32>, memref<256x256xi32, #hal.descriptor_type<storage_buffer>>


// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @conv_nhwc_dispatch_0 {
hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb">) {
  hal.executable.export @conv_nhwc layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @conv_nhwc() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2x258x514x768xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<3x3x768x256xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x256x512x256xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 258, 514, 768], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x258x514x768xf16>> -> tensor<2x258x514x768xf16>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 768, 256], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x768x256xf16>> -> tensor<3x3x768x256xf16>
        %5 = tensor.empty() : tensor<2x256x512x256xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2x256x512x256xf32>) -> tensor<2x256x512x256xf32>
        %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%3, %4 : tensor<2x258x514x768xf16>, tensor<3x3x768x256xf16>) outs(%6 : tensor<2x256x512x256xf32>) -> tensor<2x256x512x256xf32>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 256, 512, 256], strides = [1, 1, 1, 1] : tensor<2x256x512x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x256x512x256xf32>>
        return
      }
    }
  }
}

//    CHECK-LABEL: func.func @conv_nhwc
//          CHECK:   scf.for {{.*}} = %c0 to %c215 step %c1 iter_args(%[[ARG:.+]] = {{.*}}) -> (vector<2x4x1x1x4x1xf32>)
// CHECK-COUNT-16:     amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
//          CHECK:     scf.yield
// CHECK-COUNT-16:   amdgpu.mfma
//  CHECK-COUNT-8:   vector.transfer_write {{.+}} : vector<4x1xf32>, memref<2x256x512x256xf32, #hal.descriptor_type<storage_buffer>>

// -----

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb">
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#pipeline_layout = #hal.pipeline.layout<
  push_constants = 2,
  sets = [
    <0, bindings = [
      <0, storage_buffer, ReadOnly>,
      <1, storage_buffer, ReadOnly>,
      <2, storage_buffer>
    ]>
  ]>
hal.executable public @main_dispatch_expanded_matmul {
  hal.executable.variant public @rocm_hsaco_fb target(#executable_target_rocm_hsaco_fb) {
    hal.executable.export public @generic_2x1024x20x64x1280_f16 ordinal(0) layout(#pipeline_layout) attributes {
      hal.interface.bindings = [
        #hal.interface.binding<0, 0>,
        #hal.interface.binding<0, 1>,
        #hal.interface.binding<0, 2>
      ]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @generic_2x1024x20x64x1280_f16() {
        %cst = arith.constant 0.000000e+00 : f16
        %c0 = arith.constant 0 : index
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = arith.index_castui %0 : i32 to index
        %3 = arith.index_castui %1 : i32 to index
        %4 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%2) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x1024x1280xf16>>
        %5 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<20x64x1280xf16>>
        %6 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%3) : !flow.dispatch.tensor<writeonly:tensor<2x1024x20x64xf16>>
        %7 = flow.dispatch.tensor.load %4, offsets = [0, 0, 0], sizes = [2, 1024, 1280], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x1024x1280xf16>> -> tensor<2x1024x1280xf16>
        %8 = flow.dispatch.tensor.load %5, offsets = [0, 0, 0], sizes = [20, 64, 1280], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<20x64x1280xf16>> -> tensor<20x64x1280xf16>
        %9 = tensor.empty() : tensor<2x1024x20x64xf16>
        %10 = linalg.fill ins(%cst : f16) outs(%9 : tensor<2x1024x20x64xf16>) -> tensor<2x1024x20x64xf16>
        %11 = linalg.generic {
          indexing_maps = [#map, #map1, #map2],
          iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
        } ins(%7, %8 : tensor<2x1024x1280xf16>, tensor<20x64x1280xf16>)
          outs(%10 : tensor<2x1024x20x64xf16>) {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %12 = arith.mulf %in, %in_0 : f16
          %13 = arith.addf %out, %12 : f16
          linalg.yield %13 : f16
        } -> tensor<2x1024x20x64xf16>
        flow.dispatch.tensor.store %11, %6, offsets = [0, 0, 0, 0], sizes = [2, 1024, 20, 64], strides = [1, 1, 1, 1] : tensor<2x1024x20x64xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x1024x20x64xf16>>
        return
      }
    }
  }
}


//       CHECK: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUVectorDistribute workgroup_size = [128, 2, 1] subgroup_size = 64
//  CHECK-SAME:   mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>,
//  CHECK-SAME:     subgroup_m_count = 2, subgroup_n_count = 2>
//  CHECK-SAME: prefetch_shared_memory

//    CHECK-LABEL: func.func @generic_2x1024x20x64x1280_f16
// This has more than 2 iteartions. So we have prefetching enabled for this case. Due to
// prefetching, we have one iteration peeled of so upper bound is 1280 - 128 = 1152.
//          CHECK:   scf.for {{.*}} = %c0 to %c9 step %c1 iter_args({{.*}}) -> (vector<2x2x1x1x4x1xf16>)
// Each subgroup handles 2 * 2 tiles, and for each tile we accumulate 8 times
// along the K dimension. So in total 32 mfma ops.
// CHECK-COUNT-32:     amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
//          CHECK:     scf.yield %{{.+}} : vector<2x2x1x1x4x1xf16>
// CHECK-COUNT-32:   amdgpu.mfma
//  CHECK-COUNT-4:   vector.transfer_write {{.+}} : vector<4x1xf16>

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @matmul_256x256x256_f16_f32 {
hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb">) {
  hal.executable.export @matmul_256x256x256_f16_f32 layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @matmul_256x256x256_f16_f32() {
      %cst = arith.constant 0.000000e+00 : f32
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256x256xf16>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256x256xf16>>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<256x256xf32>>
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x256xf16>> -> tensor<256x256xf16>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x256xf16>> -> tensor<256x256xf16>
      %5 = tensor.empty() : tensor<256x256xf32>
      %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<256x256xf32>) -> tensor<256x256xf32>
      %7 = linalg.matmul ins(%3, %4 : tensor<256x256xf16>, tensor<256x256xf16>) outs(%6 : tensor<256x256xf32>) -> tensor<256x256xf32>
      flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : tensor<256x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<256x256xf32>>
      return
    }
  }
}
}

//       RDNA3: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUVectorDistribute workgroup_size = [64, 2, 1] subgroup_size = 32
//  RDNA3-SAME:   mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mma_layout<WMMA_F16_16x16x16_F32>,
//  RDNA3-SAME:     subgroup_m_count = 2, subgroup_n_count = 2>
//  RDNA3-SAME:   prefetch_shared_memory

//    RDNA3-LABEL: func.func @matmul_256x256x256_f16_f32
//     RDNA3-SAME:    translation_info = #[[$TRANSLATION]]
//          RDNA3:   scf.for {{.*}} = %c0 to %c256 step %c128 iter_args({{.*}}) -> (vector<2x2x8x1x1x1xf32>)
// Each subgroup handles 2 * 2 tiles, and for each tile we accumulate 8 times
// along the K dimension. So in total 32 wmma ops.
// RDNA3-COUNT-32:     amdgpu.wmma {{.*}} : vector<16xf16>, vector<16xf16>, vector<8xf32>
//          RDNA3:     scf.yield %{{.+}} : vector<2x2x8x1x1x1xf32>
//  Since each subgroup handles 2 * 2 tiles, and for each tile, each lane holds 4 values.
//  we will have 32 writes. We cannot do contiguous writes since the outputs columns has interleaved
//  thread ids.
//  RDNA3-COUNT-32:   vector.transfer_write {{.+}} {in_bounds = [true, true]} : vector<1x1xf32>, memref<256x256xf32, #hal.descriptor_type<storage_buffer>>

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @unaligned_mk_batch_matmul_64x978x1281x1281_f16_f16 {
hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb">) {
  hal.executable.export @unaligned_mk_batch_matmul_64x978x1281x1281_f16_f16 layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @unaligned_nk_batch_matmul() {
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x968x1281xf16>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x1281x1281xf16>>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<64x968x1281xf16>>
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [64, 968, 1281], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x968x1281xf16>> -> tensor<64x968x1281xf16>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [64, 1281, 1281], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x1281x1281xf16>> -> tensor<64x1281x1281xf16>
      %5 = tensor.empty() : tensor<64x968x1281xf16>
      %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<64x968x1281xf16>) -> tensor<64x968x1281xf16>
      %7 = linalg.batch_matmul ins(%3, %4 : tensor<64x968x1281xf16>, tensor<64x1281x1281xf16>) outs(%6 : tensor<64x968x1281xf16>) -> tensor<64x968x1281xf16>
      flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0], sizes = [64, 968, 1281], strides = [1, 1, 1] : tensor<64x968x1281xf16> -> !flow.dispatch.tensor<writeonly:tensor<64x968x1281xf16>>
      return
    }
  }
}
}
// Basic pipeline test to make sure it generates the instructions we expect.

// CHECK:       #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUPadAndVectorDistribute workgroup_size = [64, 1, 1] subgroup_size = 64
// CHECK-SAME:    mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>,
// CHECK-SAME:    subgroup_m_count = 1, subgroup_n_count = 1>
// CHECK-SAME:    prefetch_shared_memory

// CHECK-LABEL: func.func @unaligned_nk_batch_matmul()
// CHECK-SAME:    translation_info = #[[$TRANSLATION]]
// CHECK-DAG:     %[[RHS_SHARED:.+]] = memref.alloc() : memref<1x16x20xf16, #gpu.address_space<workgroup>>
// CHECK-DAG:     %[[RHS_SHARED_SUB:.+]] =  memref.subview %[[RHS_SHARED]][0, 0, 0] [1, 16, 16] [1, 1, 1]
// CHECK-DAG:     %[[LHS_SHARED:.+]] = memref.alloc() : memref<1x16x20xf16, #gpu.address_space<workgroup>>
// CHECK-DAG:     %[[LHS_SHARED_SUB:.+]] =  memref.subview %[[LHS_SHARED]][0, 0, 0] [1, 16, 16] [1, 1, 1]
// CHECK-DAG:     %[[LHS_GLOBAL:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<64x968x1281xf16, #hal.descriptor_type<storage_buffer>>
// CHECK-DAG:     %[[RHS_GLOBAL:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<64x1281x1281xf16, #hal.descriptor_type<storage_buffer>>
// CHECK-DAG:     %[[OUT_GLOBAL:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<64x968x1281xf16, #hal.descriptor_type<storage_buffer>>
// CHECK-DAG:     %[[LHS_GLOBAL_SUB:.+]] = memref.subview %[[LHS_GLOBAL]]
// CHECK-DAG:     %[[RHS_GLOBAL_SUB:.+]] = memref.subview %[[RHS_GLOBAL]]
// CHECK:         %[[LHS_LOAD:.+]] = vector.transfer_read %[[LHS_GLOBAL_SUB]]{{.+}} {in_bounds = [true, false, false]}
// CHECK:         %[[RHS_LOAD:.+]] = vector.transfer_read %[[RHS_GLOBAL_SUB]]{{.+}} {in_bounds = [true, false, false]}
// CHECK:         vector.transfer_write %[[LHS_LOAD]], %[[LHS_SHARED]]
// CHECK:         vector.transfer_write %[[RHS_LOAD]], %[[RHS_SHARED]]
// CHECK:         %[[RES:.+]] scf.for {{.*}} = %c0 to %c80 step %c1 iter_args({{.*}}) -> (vector<1x1x1x1x1x1x1x4x1xf16>)
// CHECK-DAG:       %[[LHS_GLOBAL_SUB:.+]] = memref.subview %[[LHS_GLOBAL]]
// CHECK-DAG:       %[[RHS_GLOBAL_SUB:.+]] = memref.subview %[[RHS_GLOBAL]]
// CHECK:           %[[LHS_LOAD:.+]] = vector.transfer_read %[[LHS_GLOBAL_SUB]]
// CHECK:           %[[RHS_LOAD:.+]] = vector.transfer_read %[[RHS_GLOBAL_SUB]]{{.+}} {in_bounds = [true, false, false]}
// CHECK:           gpu.barrier
// CHECK-DAG:       %{{.+}} = vector.transfer_read %[[LHS_SHARED]]
// CHECK-DAG:       %{{.+}} = vector.transfer_read %[[RHS_SHARED]]
// CHECK:           amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
// CHECK:           %[[TRUNCF:.+]] = arith.truncf %{{.+}} : vector<1x1x1x1x1x1x1x4x1xf32> to vector<1x1x1x1x1x1x1x4x1xf16>
// CHECK:           gpu.barrier
// CHECK:           vector.transfer_write %[[LHS_LOAD]], %[[LHS_SHARED]]
// CHECK:           vector.transfer_write %[[RHS_LOAD]], %[[RHS_SHARED]]
// CHECK:           scf.yield %[[TRUNCF]]
// CHECK:         }
// CHECK:         gpu.barrier
// CHECK:         amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
// CHECK:         %[[OUT_GLOBAL_SUB:.+]] = memref.subview %[[OUT_GLOBAL]]
// CHECK:         vector.transfer_write %{{.+}}, %[[OUT_GLOBAL_SUB]]

// -----

// This test ensures that we are generating contraction schedules does not only work on contraction,
// but also will be compatible with transfer_read layouts anchors.
// Currently the transfer_read layout anchors expects WorkgroupSize % (WgTileSize / numelPerThread) == 0.
// this test ensure that this constraint is satisfied.

// NOTE: This test is not exhaustive of all possible ways the above condition is breaking,
//       but rather is an example of a matmul shape from a model that broke our compilation heuristic.

#pipeline_layout = #hal.pipeline.layout<
  push_constants = 3,
  sets = [
    <0, bindings = [
      <0, storage_buffer, ReadOnly>,
      <1, storage_buffer>
      ]>
  ]>
hal.executable public @contract_schedule_considering_read_layout {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @contract_schedule_considering_read_layout ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @contract_schedule_considering_read_layout() {
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = hal.interface.constant.load[2] : i32
        %3 = arith.index_castui %0 : i32 to index
        %4 = arith.index_castui %1 : i32 to index
        %5 = arith.index_castui %2 : i32 to index
        %6 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%3) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x160x1536xf16>>
        %7 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%4) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x1536x1536xf16>>
        %8 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%5) : !flow.dispatch.tensor<writeonly:tensor<2x160x1536xf16>>
        %9 = flow.dispatch.tensor.load %6, offsets = [0, 0, 0], sizes = [2, 160, 1536], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x160x1536xf16>> -> tensor<2x160x1536xf16>
        %10 = flow.dispatch.tensor.load %7, offsets = [0, 0, 0], sizes = [2, 1536, 1536], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x1536x1536xf16>> -> tensor<2x1536x1536xf16>
        %11 = tensor.empty() : tensor<2x160x1536xf16>
        %12 = linalg.fill ins(%cst : f16) outs(%11 : tensor<2x160x1536xf16>) -> tensor<2x160x1536xf16>
        %13 = linalg.batch_matmul ins(%9, %10 : tensor<2x160x1536xf16>, tensor<2x1536x1536xf16>) outs(%12 : tensor<2x160x1536xf16>) -> tensor<2x160x1536xf16>
        flow.dispatch.tensor.store %13, %8, offsets = [0, 0, 0], sizes = [2, 160, 1536], strides = [1, 1, 1] : tensor<2x160x1536xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x160x1536xf16>>
        return
      }
    }
  }
}
// Basic pipeline test to make sure it generates the instructions we expect.

// CHECK:       #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 64
// CHECK-SAME:    mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>,
// CHECK-SAME:    subgroup_m_count = 1, subgroup_n_count = 4>
// CHECK-SAME:    prefetch_shared_memory

// CHECK-LABEL: func.func @contract_schedule_considering_read_layout()
// CHECK-SAME:    translation_info = #[[$TRANSLATION]]
// CHECK-DAG:     %[[RHS_SHARED:.+]] = memref.alloc() : memref<128x132xf16, #gpu.address_space<workgroup>>
// CHECK-DAG:     %[[RHS_SHARED_SUB:.+]] =  memref.subview %[[RHS_SHARED]][0, 0] [128, 128] [1, 1]
// CHECK-DAG:     %[[LHS_SHARED:.+]] = memref.alloc() : memref<16x132xf16, #gpu.address_space<workgroup>>
// CHECK-DAG:     %[[LHS_SHARED_SUB:.+]] =  memref.subview %[[LHS_SHARED]][0, 0] [16, 128] [1, 1]
// CHECK:   scf.for {{.*}} = %c0 to %c11 step %c1 iter_args(%[[ARG:.+]] = {{.*}}) -> (vector<1x2x1x1x4x1xf16>)
// CHECK-COUNT-16:     amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
// CHECK:     scf.yield
// CHECK-COUNT-16:   amdgpu.mfma

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer>]>]>
hal.executable private @attention_20x4096x64x4096x64 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @attention_20x4096x64x4096x64 ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @attention_20x4096x64x4096x64() {
        %cst = arith.constant 1.250000e-01 : f16
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<20x4096x64xf16>>
        %4 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [20, 4096, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>> -> tensor<20x4096x64xf16>
        %5 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [20, 4096, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>> -> tensor<20x4096x64xf16>
        %6 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [20, 4096, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>> -> tensor<20x4096x64xf16>
        %7 = tensor.empty() : tensor<20x4096x64xf16>
        %8 = iree_linalg_ext.attention ins(%4, %5, %6, %cst : tensor<20x4096x64xf16>, tensor<20x4096x64xf16>, tensor<20x4096x64xf16>, f16) outs(%7 : tensor<20x4096x64xf16>) -> tensor<20x4096x64xf16>
        flow.dispatch.tensor.store %8, %3, offsets = [0, 0, 0], sizes = [20, 4096, 64], strides = [1, 1, 1] : tensor<20x4096x64xf16> -> !flow.dispatch.tensor<writeonly:tensor<20x4096x64xf16>>
        return
      }
    }
  }
}

// Basic test to make sure we can handle attention

// CHECK:       #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUVectorDistribute workgroup_size = [64, 1, 1] subgroup_size = 64
// CHECK-SAME:    mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>,
// CHECK-SAME:    subgroup_m_count = 1, subgroup_n_count = 1>
// Prefetching is disabled for attention for now
// CHECK-NOT:     prefetch_shared_memory

// CHECK-LABEL: func.func @attention_20x4096x64x4096x64()
// CHECK-SAME:    translation_info = #[[$TRANSLATION]]

// CHECK: scf.for %{{.*}} = %c0 to %c4096 step %c32
// CHECK-SAME: -> (vector<4x1x4xf32>, vector<4x4x1x1x4x1xf16>, vector<4x1x4xf32>)
// CHECK-COUNT-64:  amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
// CHECK: scf.yield
