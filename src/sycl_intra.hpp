/*
 * Copyright (c) 2021, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 3-Clause Clear License
 * and the Alliance for Open Media Patent License 1.0. If the BSD 3-Clause Clear
 * License was not distributed with this source code in the LICENSE file, you
 * can obtain it at aomedia.org/license/software-license/bsd-3-c-c/.  If the
 * Alliance for Open Media Patent License 1.0 was not distributed with this
 * source code in the PATENTS file, you can obtain it at
 * aomedia.org/license/patent-license/.
 */

#ifndef AVM_DSP_SYCL_SYCL_INTRA_HPP_
#define AVM_DSP_SYCL_SYCL_INTRA_HPP_

#ifdef HAVE_SYCL

#include <cstdint>
#include <cstddef>

namespace avm {
namespace sycl {

// DC prediction
void intra_pred_dc_4x4(const uint8_t* ref, uint8_t* dst, int stride);
void intra_pred_dc_8x8(const uint8_t* ref, uint8_t* dst, int stride);
void intra_pred_dc_16x16(const uint8_t* ref, uint8_t* dst, int stride);
void intra_pred_dc_32x32(const uint8_t* ref, uint8_t* dst, int stride);

// Horizontal prediction
void intra_pred_h_4x4(const uint8_t* ref, uint8_t* dst, int stride);
void intra_pred_h_8x8(const uint8_t* ref, uint8_t* dst, int stride);
void intra_pred_h_16x16(const uint8_t* ref, uint8_t* dst, int stride);
void intra_pred_h_32x32(const uint8_t* ref, uint8_t* dst, int stride);

// Vertical prediction
void intra_pred_v_4x4(const uint8_t* ref, uint8_t* dst, int stride);
void intra_pred_v_8x8(const uint8_t* ref, uint8_t* dst, int stride);
void intra_pred_v_16x16(const uint8_t* ref, uint8_t* dst, int stride);
void intra_pred_v_32x32(const uint8_t* ref, uint8_t* dst, int stride);

}  // namespace sycl
}  // namespace avm

#endif  // HAVE_SYCL
#endif  // AVM_DSP_SYCL_SYCL_INTRA_HPP_
