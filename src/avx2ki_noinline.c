/*************************************************************************************
   AVX2KI Noinline Wrappers — Override library's slow _mm_blendv_epi8 and _mm_movemask_epi8

   These two intrinsics consume 50%+ of smithWaterman time when resolved from
   libavx2neon.so. The library's implementations are ~200 instructions each (extracts
   bytes individually with stack canary). Our wrappers are 3-5 NEON instructions.

   Why a separate .c file instead of inline/noinline/macro in avx2ki_inline.h?
   - always_inline: causes massive caller code bloat (smithWaterman128_16: 2780→7356B)
     due to register pressure and optimization path changes when compiler sees
     blendv/movemask implementations inline in smithWaterman
   - noinline function in header: GCC 10.3.1 ICE on static __attribute__((noinline))
     with __m128i union parameters inside extern "C" blocks
   - noinline outside extern "C": conflicts with avx.h's later extern "C" declarations
     (different linkage for same function name = compilation error)
   - macro: same bloat as always_inline (statement expression expands inline)
   - separate .c file: compiled independently, no caller bloat, no linkage conflicts

   How it works:
   - avx.h declares _mm_blendv_epi8 and _mm_movemask_epi8 as extern "C" functions
   - libavx2neon.so provides the (slow) implementations
   - This .c file provides our own extern "C" implementations with same signatures
   - The linker resolves to our definitions because they appear before -lavx2neon
   - Result: no PLT call to library, no caller bloat, just a direct bl to our tiny wrapper
*************************************************************************************/

#include <arm_neon.h>
#include "avx2ki_type.h"  /* __m128i union definition */

__m128i _mm_blendv_epi8(__m128i a, __m128i b, __m128i mask)
{
    /*
     * _mm_blendv_epi8(a, b, mask): for each byte lane,
     *   if mask byte's sign bit is 1 → select from b
     *   if mask byte's sign bit is 0 → select from a
     *
     * Same as bwa-mem2's own SSE2 fallback in bandedSWA.cpp:
     *   _mm_or_si128(_mm_andnot_si128(mask, a), _mm_and_si128(mask, b))
     * Translates to: vbic + vand + vorr = 3 NEON instructions.
     */
    __m128i r;
    r.vect_u8 = vorrq_u8(vbicq_u8(a.vect_u8, mask.vect_u8),
                          vandq_u8(b.vect_u8, mask.vect_u8));
    return r;
}

int _mm_movemask_epi8(__m128i a)
{
    /*
     * _mm_movemask_epi8(a): extract the sign bit of each of the 16 bytes,
     * packing them into the low 16 bits of an int.
     *
     * Same algorithm as sse2neon:
     * 1. Shift each byte right by 7 → each byte is 0 or 1
     * 2. Shift each byte left by its bit position (0-7 per half) using vshlq_u8
     * 3. Horizontal add each 8-byte half → bits [7:0] and [15:8]
     */
    static const int8_t shift_table[16] = {0, 1, 2, 3, 4, 5, 6, 7,
                                           0, 1, 2, 3, 4, 5, 6, 7};
    uint8x16_t msbs = vshrq_n_u8(a.vect_u8, 7);
    int8x16_t shifts = vld1q_s8(shift_table);
    uint8x16_t positioned = vshlq_u8(msbs, shifts);
    return vaddv_u8(vget_low_u8(positioned)) |
           (vaddv_u8(vget_high_u8(positioned)) << 8);
}