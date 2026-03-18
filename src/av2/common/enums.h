// av2/common/enums.h - Minimal stub for standalone build
#ifndef AV2_COMMON_ENUMS_H_
#define AV2_COMMON_ENUMS_H_

// Transform types
typedef enum {
  DCT_DCT = 0,    // DCT in horizontal and vertical
  ADST_DCT = 1,   // ADST in vertical, DCT in horizontal
  DCT_ADST = 2,   // DCT in vertical, ADST in horizontal
  ADST_ADST = 3,  // ADST in both directions
  FLIPADST_DCT = 4,
  DCT_FLIPADST = 5,
  FLIPADST_FLIPADST = 6,
  ADST_FLIPADST = 7,
  FLIPADST_ADST = 8,
  IDTX = 9,
  V_DCT = 10,
  H_DCT = 11,
  V_ADST = 12,
  H_ADST = 13,
  V_FLIPADST = 14,
  H_FLIPADST = 15,
  TX_TYPES = 16
} TX_TYPE;

// Transform sizes
typedef enum {
  TX_4X4 = 0,
  TX_8X8 = 1,
  TX_16X16 = 2,
  TX_32X32 = 3,
  TX_64X64 = 4,
  TX_SIZES = 5
} TX_SIZE;

// Block sizes
typedef enum {
  BLOCK_4X4 = 0,
  BLOCK_4X8 = 1,
  BLOCK_8X4 = 2,
  BLOCK_8X8 = 3,
  BLOCK_8X16 = 4,
  BLOCK_16X8 = 5,
  BLOCK_16X16 = 6,
  BLOCK_16X32 = 7,
  BLOCK_32X16 = 8,
  BLOCK_32X32 = 9,
  BLOCK_32X64 = 10,
  BLOCK_64X32 = 11,
  BLOCK_64X64 = 12,
  BLOCK_SIZES = 13
} BLOCK_SIZE;

// Prediction modes
typedef enum {
  DC_PRED = 0,
  V_PRED = 1,
  H_PRED = 2,
  D45_PRED = 3,
  D135_PRED = 4,
  D113_PRED = 5,
  D157_PRED = 6,
  D203_PRED = 7,
  D67_PRED = 8,
  SMOOTH_PRED = 9,
  SMOOTH_V_PRED = 10,
  SMOOTH_H_PRED = 11,
  PAETH_PRED = 12,
  NEARESTMV = 13,
  NEARMV = 14,
  GLOBALMV = 15,
  NEWMV = 16,
  MB_MODE_COUNT = 17
} PREDICTION_MODE;

#endif  // AV2_COMMON_ENUMS_H_
