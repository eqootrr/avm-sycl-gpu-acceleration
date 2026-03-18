/*
 * Copyright (c) 2021, Alliance for Open Media. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 3-Clause Clear License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 3-Clause Clear
 * License was not distributed with this source code in the LICENSE file, you
 * can obtain it at aomedia.org/license/software-license/bsd-3-c-c/.  If the
 * Alliance for Open Media Patent License 1.0 was not distributed with this
 * source code in the PATENTS file, you can obtain it at
 * aomedia.org/license/patent-license/.
 */

#ifndef AVM_DSP_SYCL_SYCL_WRAPPER_HPP_
#define AVM_DSP_SYCL_SYCL_WRAPPER_HPP_

#include "avm_config.h"

#ifdef HAVE_SYCL

#include "sycl_txfm.hpp"
#include "sycl_me.hpp"
#include "sycl_lpf.hpp"
#include "sycl_intra.hpp"

namespace avm {
namespace sycl {

// Check if SYCL should be used (called by RTCD)
inline bool should_use_sycl() {
    return SYCLContext::instance().is_available() &&
           SYCLContext::instance().is_gpu();
}

// RTCD function registration
void register_sycl_functions();

}  // namespace sycl
}  // namespace avm

#endif  // HAVE_SYCL
#endif  // AVM_DSP_SYCL_SYCL_WRAPPER_HPP_
