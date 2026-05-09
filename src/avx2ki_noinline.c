/*************************************************************************************
   AVX2KI Noinline Wrappers — NEON-native versions of blendv and movemask

   These intrinsics consume 50%+ of smithWaterman time when resolved from
   libavx2neon.so. The library's implementations are ~200 instructions each.

   CRITICAL PERFORMANCE ISSUE: __m128i union is passed via GENERAL registers (x0-x7)
   under AAPCS64, not vector registers. This means each call to a function taking
   __m128i requires 6 fmov instructions (GPR→NEON) for arguments + 2 fmov (NEON→GPR)
   for return = 8 wasted instructions per call.

   SOLUTION: simd_compat.h defines _mm_blendv_epi8 and _mm_movemask_epi8 as macros
   that extract the NEON vector member (.vect_u8) and call native NEON functions
   (_blendv_epi8_neon, _movemask_epi8_neon) that take uint8x16_t arguments (passed
   via vector registers v0-v7 under AAPCS64). This eliminates ALL fmov overhead
   while keeping the function call boundary that the compiler needs for good
   instruction scheduling.

   Previous failed approaches:
   - always_inline: causes massive caller code bloat (smithWaterman128_16: 2780→7356B)
   - macro (decompose blendv to andnot+and+or): severe regression (+54%)
   - noinline with __m128i args: 8 fmov per call (current baseline)
   - NEON-native args: zero fmov overhead, preserves call boundary ← this approach
*************************************************************************************/

#include <arm_neon.h>
#include "avx2ki_type.h"  /* __m128i union definition */

/*
 * _blendv_epi8_neon: native NEON vector version of _mm_blendv_epi8.
 * Takes uint8x16_t arguments (passed in vector registers v0-v7),
 * returns uint8x16_t (in v0). Zero fmov overhead.
 *
 * Uses ARM's BIT/BSL instruction which is a single-instruction
 * blendv: result = (b & mask) | (a & ~mask).
 */
uint8x16_t _blendv_epi8_neon(uint8x16_t a, uint8x16_t b, uint8x16_t mask)
{
    return vbslq_u8(mask, b, a);
}

/*
 * _movemask_epi8_neon: native NEON vector version of _mm_movemask_epi8.
 * Takes uint8x16_t argument (passed in vector register v0),
 * returns int. Zero fmov overhead on input.
 *
 * Algorithm: extract sign bits of each byte into a 16-bit integer.
 * 1. Shift each byte right by 7 → each byte is 0 or 1
 * 2. Shift each byte left by its bit position (0-7 per half) using vshlq_u8
 * 3. Horizontal add each 8-byte half → bits [7:0] and [15:8]
 */
int _movemask_epi8_neon(uint8x16_t a)
{
    static const int8_t shift_table[16] = {0, 1, 2, 3, 4, 5, 6, 7,
                                           0, 1, 2, 3, 4, 5, 6, 7};
    uint8x16_t msbs = vshrq_n_u8(a, 7);
    int8x16_t shifts = vld1q_s8(shift_table);
    uint8x16_t positioned = vshlq_u8(msbs, shifts);
    return vaddv_u8(vget_low_u8(positioned)) |
           (vaddv_u8(vget_high_u8(positioned)) << 8);
}
