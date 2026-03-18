// avm_dsp/loopfilter.h - Minimal stub for standalone build
#ifndef AVM_DSP_LOOPFILTER_H_
#define AVM_DSP_LOOPFILTER_H_

#include <cstdint>

// Filter level
static const int MAX_LOOP_FILTER = 63;

// Filter mask
typedef uint8_t LOOP_FILTER_MASK;

// Edge direction
typedef enum {
  VERT_EDGE = 0,
  HORZ_EDGE = 1,
  NUM_EDGE_DIRS = 2
} EDGE_DIR;

// Filter parameters
struct LoopFilterParams {
  int filter_level[2];  // [0] = luma, [1] = chroma
  int sharpness_level;
  int mode_deltas[2];
};

#endif  // AVM_DSP_LOOPFILTER_H_
