/*************************************************************************************
                           The MIT License

   BWA-MEM2  (Sequence alignment using Burrows-Wheeler Transform),
   Copyright (C) 2019  Intel Corporation, Heng Li.

   Permission is hereby granted, free of charge, to any person obtaining
   a copy of this software and associated documentation files (the
   "Software"), to deal in the Software without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Software, and to
   permit persons to whom the Software is furnished to do so, subject to
   the following conditions:

   The above copyright notice and this permission notice shall be
   included in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
   BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
   ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.

   ARM compatibility layer for bwa-mem2 on ARM64/Kunpeng 920
   Uses AVX2KI (Kunpeng System Library) for AVX2-to-NEON translation
*****************************************************************************************/

#ifndef SIMD_COMPAT_H
#define SIMD_COMPAT_H

#if defined(__ARM_NEON) || defined(__aarch64__)

/*
 * AVX2KI Inline Shim Optimization
 *
 * Strategy: include ONLY avx2ki_type.h (type definitions) + our inline shim
 * for hot intrinsics, then include avx.h for the remaining intrinsics that
 * must use the library (srli/slli/movemask/blendv/extract).
 *
 * avx2ki_type.h: defines __m128i union with NEON vector members
 * avx2ki_inline.h: static inline NEON wrappers for ~25 hot intrinsics
 * avx.h: extern "C" declarations for remaining intrinsics (linked from libavx2neon.so)
 *
 * Since avx2ki_inline.h uses static inline (internal linkage), there is no
 * conflict with avx.h's extern "C" declarations — the compiler resolves
 * the static inline first for inlining, and the extern symbol is unused.
 */
#include "avx2ki_type.h"
#include "avx2ki_inline.h"
#include "avx.h"

/*
 * Macro overrides: redirect _mm_srli/slli/extract to specialized inline wrappers.
 * These MUST come after avx.h (which declares the extern "C" prototypes),
 * otherwise the macros would corrupt avx.h's declarations.
 * After avx.h is processed, these #define substitutions redirect all subsequent
 * calls (in ksw.cpp, kswv.cpp, etc.) to the specialized _mm_*_N() inlines
 * defined in avx2ki_inline.h, eliminating PLT calls to libavx2neon.so.
 */
#define _mm_srli_si128(a, imm) _mm_srli_si128_##imm(a)
#define _mm_slli_si128(a, imm) _mm_slli_si128_##imm(a)
#define _mm_extract_epi16(a, imm) _mm_extract_epi16_##imm(a)

/*
 * Forward declaration of NEON-native wrapper functions — defined in avx2ki_noinline.c.
 * _blendv_epi8_neon takes native NEON uint8x16_t arguments (passed in vector
 * registers v0-v7 under AAPCS64) instead of __m128i union (passed in general
 * registers). This eliminates 8 fmov instructions per blendv call.
 *
 * _movemask_epi8_neon is kept for linkage compatibility but is NOT called at
 * runtime — _mm_movemask_epi8 is now an inline macro (see below).
 */
extern "C" uint8x16_t _blendv_epi8_neon(uint8x16_t a, uint8x16_t b, uint8x16_t mask);
extern "C" int _movemask_epi8_neon(uint8x16_t a);

/*
 * _mm_blendv_epi8 macro override: redirect to _blendv_epi8_neon() which
 * takes native NEON vector arguments instead of __m128i union.
 *
 * CRITICAL: Under AAPCS64, __m128i union is passed via GENERAL registers
 * (x0-x7), requiring 6 fmov (GPR→NEON) + 2 fmov (NEON→GPR) per call =
 * 8 wasted instructions. With ~13 blendv calls per SW row, that's ~104
 * wasted fmov instructions.
 *
 * By extracting the .vect_u8 member BEFORE the call and passing it to a
 * function that takes uint8x16_t (passed in vector registers v0-v7), we
 * eliminate ALL fmov overhead while keeping the function call boundary
 * that the compiler needs for good instruction scheduling in BSW.
 *
 * The GCC statement expression ({...}) is used to create a temporary
 * __m128i result and assign the NEON return value to its .vect_u8 member.
 */
#define _mm_blendv_epi8(a, b, mask) __extension__({ \
    __m128i _r; \
    _r.vect_u8 = _blendv_epi8_neon((a).vect_u8, (b).vect_u8, (mask).vect_u8); \
    _r; \
})

/*
 * _mm_movemask_epi8 macro override: INLINE expansion using NEON intrinsics.
 *
 * Previous approach (_movemask_epi8_neon function call) eliminated fmov overhead
 * but introduced a worse problem: the function call boundary forces the compiler
 * to spill/restore vector registers around the call site. In ksw_u8's lazy-F loop,
 * this caused 6 extra stack spill/restore instructions per iteration, making the
 * loop significantly slower than sse2neon (which inlines movemask).
 *
 * This inline macro expands to the same NEON instruction sequence that
 * _movemask_epi8_neon uses, but without the function call boundary. The compiler
 * can keep all vector values in registers across the movemask operation, avoiding
 * stack spills. With only ~1-2 movemask calls per inner loop iteration, the code
 * bloat risk is low (unlike blendv's ~33 calls/row which caused +54% regression).
 *
 * Algorithm: extract sign bits of each of the 16 bytes into a 16-bit integer.
 * 1. Shift each byte right by 7 → each byte is 0 or 1
 * 2. Shift each byte left by its bit position (0-7 per half) using vshlq_u8
 * 3. Horizontal add each 8-byte half → bits [7:0] and [15:8]
 * 4. Combine halves into final 16-bit result
 */
#define _mm_movemask_epi8(a) __extension__({ \
    uint8x16_t _a = (a).vect_u8; \
    const int8_t _shift[16] = {0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7}; \
    uint8x16_t _msbs = vshrq_n_u8(_a, 7); \
    int8x16_t _shifts = vld1q_s8(_shift); \
    uint8x16_t _pos = vshlq_u8(_msbs, _shifts); \
    int _r = (int)vaddv_u8(vget_low_u8(_pos)) | \
             ((int)vaddv_u8(vget_high_u8(_pos)) << 8); \
    _r; \
})

/*
 * ARM NEON does not have _mm_malloc/_mm_free.
 * Use posix_memalign / free as portable alternatives.
 *
 * Note: _mm_malloc(size, align) takes (size, alignment)
 *       posix_memalign(&ptr, alignment, size) takes (&ptr, alignment, size)
 *       The parameter ORDER is different!
 */

#include <cstdlib>
#include <cerrno>

static inline void* aligned_alloc_compat(size_t alignment, size_t size)
{
    void* ptr = nullptr;
    /* posix_memalign requires alignment to be a power of 2 and a multiple of sizeof(void*) */
    if (alignment < sizeof(void*))
        alignment = sizeof(void*);
    int ret = posix_memalign(&ptr, alignment, size);
    if (ret != 0) {
        errno = ret;
        return nullptr;
    }
    return ptr;
}

/* Provide _mm_malloc / _mm_free compatible wrappers for ARM */
#define _mm_malloc(size, align) aligned_alloc_compat((align), (size))
#define _mm_free(ptr) free((ptr))

/*
 * Cache line size for Kunpeng 920: 128 bytes
 * Override if not already defined
 */
#ifndef CACHE_LINE_BYTES
#define CACHE_LINE_BYTES 128
#endif

/*
 * SIMD width definitions for ARM NEON (128-bit, equivalent to SSE4.1)
 * These should match the SSE2 path: 16 lanes of int8, 8 lanes of int16
 * The AVX2KI_FLAGS in Makefile define __SSE2__=1, so the existing
 * SIMD_WIDTH8=16 / SIMD_WIDTH16=8 path in bandedSWA.h will be taken.
 */

/*
 * _mm_countbits_64 equivalent for ARM
 * __builtin_popcountl works on both x86 and ARM with GCC/Clang
 * FMI_search.h already has a GCC/Clang-specific definition that works,
 * so we don't need to redefine it here.
 */

/*
 * Prefetch hint constants for ARM.
 * These are normally defined in xmmintrin.h on x86.
 * AVX2KI may not define all of them, so we define them here.
 */
#ifndef _MM_HINT_T0
#define _MM_HINT_T0 3
#endif
#ifndef _MM_HINT_T1
#define _MM_HINT_T1 2
#endif
#ifndef _MM_HINT_T2
#define _MM_HINT_T2 1
#endif
#ifndef _MM_HINT_NTA
#define _MM_HINT_NTA 0
#endif

/*
 * Override AVX2KI's _mm_prefetch with a const void* version.
 * avx2ki.h may define _mm_prefetch(char const*, int) but x86 intrinsics
 * accept const void*, and bwa-mem2 passes non-char pointers directly.
 */
#ifdef _mm_prefetch
#undef _mm_prefetch
#endif
#define _mm_prefetch(p, i) __builtin_prefetch((p), 0, (i) == _MM_HINT_T0 ? 3 : ((i) == _MM_HINT_T1 ? 2 : ((i) == _MM_HINT_T2 ? 1 : 0)))

/*
 * __rdtsc() replacement for ARM64
 * On x86, __rdtsc() reads the Time Stamp Counter (TSC).
 * On ARM64, we use the generic timer register (CNTVCT_EL0).
 * This is used for profiling only and does not affect correctness.
 */
#if defined(__aarch64__)
static inline uint64_t __rdtsc(void)
{
    uint64_t val;
    __asm__ volatile ("mrs %0, cntvct_el0" : "=r" (val));
    return val;
}
#endif

#else
/* x86 path: include full intrinsics header for __rdtsc and all SIMD ops */
#include <immintrin.h>
#endif /* __ARM_NEON || __aarch64__ */

#endif /* SIMD_COMPAT_H */