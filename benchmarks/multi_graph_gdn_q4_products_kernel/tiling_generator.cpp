// SPDX-License-Identifier: Apache-2.0
// Offline fixed-shape tiling generation. This executable uses no ACL/NPU APIs.
#include <iostream>
#include "tiling/matmul/matmul_tiling.h"
#include "kernel_tiling/kernel_tiling.h"

int main() {
  matmul_tiling::PlatformInfo platform{};
  platform.socVersion = platform_ascendc::SocVersion::ASCEND910B;
  platform.l1Size = 512 * 1024;
  platform.l0CSize = 128 * 1024;
  platform.ubSize = 192 * 1024;
  platform.l0ASize = platform.l0BSize = 64 * 1024;
  matmul_tiling::MatmulApiTiling generator(platform);
  generator.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BF16,
                     false);
  generator.SetBType(matmul_tiling::TPosition::TSCM, matmul_tiling::CubeFormat::NZ, matmul_tiling::DataType::DT_BF16,
                     true);
  generator.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
  generator.SetOrgShape(384, 8, 128);
  generator.SetShape(384, 8, 128);
  generator.SetFixSplit(128, 16, 128);
  generator.SetBufferSpace(256 * 1024, 128 * 1024, -1);
  generator.EnableBias(false);
  AscendC::tiling::TCubeTiling tiling{};
  if (generator.GetTiling(tiling) != 0) {
    std::cerr << "fixed Q4 products tiling failed\n";
    return 1;
  }
  std::cout << "// Offline CANN tiler output for M384 N8 K128, A GM ND, B L1 NZ transposed.\n"
               "#pragma once\n"
               "__aicore__ inline void FillProductsTiling(AscendC::tiling::TCubeTiling& tiling) {\n";
#define EMIT(field) std::cout << "  tiling." #field " = " << tiling.field << ";\n";
  EMIT(usedCoreNum)
  EMIT(M)
  EMIT(N)
  EMIT(Ka)
  EMIT(Kb)
  EMIT(singleCoreM)
  EMIT(singleCoreN)
  EMIT(singleCoreK)
  EMIT(baseM)
  EMIT(baseN)
  EMIT(baseK)
  EMIT(depthA1)
  EMIT(depthB1)
  EMIT(stepM)
  EMIT(stepN)
  EMIT(isBias)
  EMIT(transLength)
  EMIT(iterateOrder)
  EMIT(shareMode)
  EMIT(shareL1Size)
  EMIT(shareL0CSize)
  EMIT(shareUbSize)
  EMIT(batchM)
  EMIT(batchN)
  EMIT(singleBatchM)
  EMIT(singleBatchN)
  EMIT(stepKa)
  EMIT(stepKb)
  EMIT(depthAL1CacheUB)
  EMIT(depthBL1CacheUB)
  EMIT(dbL0A)
  EMIT(dbL0B)
  EMIT(dbL0C)
  EMIT(ALayoutInfoB)
  EMIT(ALayoutInfoS)
  EMIT(ALayoutInfoN)
  EMIT(ALayoutInfoG)
  EMIT(ALayoutInfoD)
  EMIT(BLayoutInfoB)
  EMIT(BLayoutInfoS)
  EMIT(BLayoutInfoN)
  EMIT(BLayoutInfoG)
  EMIT(BLayoutInfoD)
  EMIT(CLayoutInfoB)
  EMIT(CLayoutInfoS1)
  EMIT(CLayoutInfoN)
  EMIT(CLayoutInfoG)
  EMIT(CLayoutInfoS2)
  EMIT(BatchNum)
  EMIT(mxTypePara)
#undef EMIT
  std::cout << "}\n";
}
