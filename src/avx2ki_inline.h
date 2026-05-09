/*************************************************************************************
   AVX2KI Inline Shim — Map hot SSE intrinsics to native NEON

   This header provides static inline NEON translations for the most frequently
   called SSE intrinsics in bwa-mem2's smithWaterman128_16 hotspot.

   Strategy:
   - simd_compat.h includes avx2ki_type.h (type definitions), then this header,
     then avx.h (remaining extern "C" declarations for non-inlined intrinsics)
   - This header uses extern "C" { static inline ... } for C linkage compatibility
     with avx.h's later re-declarations, while static provides internal linkage
     (no multiple definition errors at link time)
   - For complex intrinsics that can't be 1:1 mapped (srli/slli/extract),
     avx.h provides extern "C" declarations that resolve to libavx2neon.so
   - For hot intrinsics that cause code bloat when inlined (blendv/movemask),
     we use static __attribute__((noinline)) — tiny local wrapper, no caller bloat

   Type compatibility: AVX2KI's __m128i is a union containing NEON vector members
   (e.g., int8x16_t vect_s8, int16x8_t vect_s16), so direct NEON operations on
   these members produce binary-compatible results.
*****************************************************************************************/

#ifndef AVX2KI_INLINE_H
#define AVX2KI_INLINE_H

#if defined(__ARM_NEON) || defined(__aarch64__)

#include <arm_neon.h>

/*
 * extern "C" + static inline: C linkage (compatible with avx.h) + internal linkage
 * (no ODR violations). The compiler inlines these; the library symbols are unused.
 */
#ifdef __cplusplus
extern "C" {
#endif

/* ========== Load/Store ========== */

static inline __attribute__((always_inline)) __m128i _mm_load_si128(const __m128i *p)
{
    __m128i r;
    r.vect_s8 = vld1q_s8((const int8_t *)p);
    return r;
}

static inline __attribute__((always_inline)) void _mm_store_si128(__m128i *p, __m128i a)
{
    vst1q_s8((int8_t *)p, a.vect_s8);
}

static inline __attribute__((always_inline)) __m128i _mm_setzero_si128(void)
{
    __m128i r;
    r.vect_s8 = vdupq_n_s8(0);
    return r;
}

/* ========== Set/Broadcast ========== */

static inline __attribute__((always_inline)) __m128i _mm_set1_epi8(char x)
{
    __m128i r;
    r.vect_s8 = vdupq_n_s8((signed char)x);
    return r;
}

static inline __attribute__((always_inline)) __m128i _mm_set1_epi16(short x)
{
    __m128i r;
    r.vect_s16 = vdupq_n_s16(x);
    return r;
}

static inline __attribute__((always_inline)) __m128i _mm_set1_epi32(int x)
{
    __m128i r;
    r.vect_s32 = vdupq_n_s32(x);
    return r;
}

/* ========== Arithmetic — 16-bit lanes ========== */

static inline __attribute__((always_inline)) __m128i _mm_add_epi16(__m128i a, __m128i b)
{
    __m128i r;
    r.vect_s16 = vaddq_s16(a.vect_s16, b.vect_s16);
    return r;
}

static inline __attribute__((always_inline)) __m128i _mm_sub_epi16(__m128i a, __m128i b)
{
    __m128i r;
    r.vect_s16 = vsubq_s16(a.vect_s16, b.vect_s16);
    return r;
}

static inline __attribute__((always_inline)) __m128i _mm_adds_epi16(__m128i a, __m128i b)
{
    __m128i r;
    r.vect_s16 = vqaddq_s16(a.vect_s16, b.vect_s16);
    return r;
}

static inline __attribute__((always_inline)) __m128i _mm_subs_epi16(__m128i a, __m128i b)
{
    __m128i r;
    r.vect_s16 = vqsubq_s16(a.vect_s16, b.vect_s16);
    return r;
}

static inline __attribute__((always_inline)) __m128i _mm_subs_epu16(__m128i a, __m128i b)
{
    __m128i r;
    r.vect_u16 = vqsubq_u16(a.vect_u16, b.vect_u16);
    return r;
}

/* ========== Arithmetic — 8-bit lanes ========== */

static inline __attribute__((always_inline)) __m128i _mm_add_epi8(__m128i a, __m128i b)
{
    __m128i r;
    r.vect_s8 = vaddq_s8(a.vect_s8, b.vect_s8);
    return r;
}

static inline __attribute__((always_inline)) __m128i _mm_sub_epi8(__m128i a, __m128i b)
{
    __m128i r;
    r.vect_s8 = vsubq_s8(a.vect_s8, b.vect_s8);
    return r;
}

static inline __attribute__((always_inline)) __m128i _mm_adds_epu8(__m128i a, __m128i b)
{
    __m128i r;
    r.vect_u8 = vqaddq_u8(a.vect_u8, b.vect_u8);
    return r;
}

static inline __attribute__((always_inline)) __m128i _mm_subs_epu8(__m128i a, __m128i b)
{
    __m128i r;
    r.vect_u8 = vqsubq_u8(a.vect_u8, b.vect_u8);
    return r;
}

/* ========== Min/Max ========== */

static inline __attribute__((always_inline)) __m128i _mm_max_epi16(__m128i a, __m128i b)
{
    __m128i r;
    r.vect_s16 = vmaxq_s16(a.vect_s16, b.vect_s16);
    return r;
}

static inline __attribute__((always_inline)) __m128i _mm_min_epi16(__m128i a, __m128i b)
{
    __m128i r;
    r.vect_s16 = vminq_s16(a.vect_s16, b.vect_s16);
    return r;
}

static inline __attribute__((always_inline)) __m128i _mm_max_epu8(__m128i a, __m128i b)
{
    __m128i r;
    r.vect_u8 = vmaxq_u8(a.vect_u8, b.vect_u8);
    return r;
}

static inline __attribute__((always_inline)) __m128i _mm_min_epu8(__m128i a, __m128i b)
{
    __m128i r;
    r.vect_u8 = vminq_u8(a.vect_u8, b.vect_u8);
    return r;
}

static inline __attribute__((always_inline)) __m128i _mm_max_epu16(__m128i a, __m128i b)
{
    __m128i r;
    r.vect_u16 = vmaxq_u16(a.vect_u16, b.vect_u16);
    return r;
}

static inline __attribute__((always_inline)) __m128i _mm_min_epu16(__m128i a, __m128i b)
{
    __m128i r;
    r.vect_u16 = vminq_u16(a.vect_u16, b.vect_u16);
    return r;
}

/* ========== Compare — 16-bit lanes ========== */

static inline __attribute__((always_inline)) __m128i _mm_cmpeq_epi16(__m128i a, __m128i b)
{
    __m128i r;
    r.vect_u16 = vceqq_s16(a.vect_s16, b.vect_s16);
    return r;
}

static inline __attribute__((always_inline)) __m128i _mm_cmpgt_epi16(__m128i a, __m128i b)
{
    __m128i r;
    r.vect_u16 = vcgtq_s16(a.vect_s16, b.vect_s16);
    return r;
}

/* ========== Compare — 8-bit lanes ========== */

static inline __attribute__((always_inline)) __m128i _mm_cmpeq_epi8(__m128i a, __m128i b)
{
    __m128i r;
    r.vect_u8 = vceqq_s8(a.vect_s8, b.vect_s8);
    return r;
}

static inline __attribute__((always_inline)) __m128i _mm_cmpgt_epi8(__m128i a, __m128i b)
{
    __m128i r;
    r.vect_u8 = vcgtq_s8(a.vect_s8, b.vect_s8);
    return r;
}

/* ========== Bitwise Logic ========== */

static inline __attribute__((always_inline)) __m128i _mm_and_si128(__m128i a, __m128i b)
{
    __m128i r;
    r.vect_s8 = vandq_s8(a.vect_s8, b.vect_s8);
    return r;
}

static inline __attribute__((always_inline)) __m128i _mm_or_si128(__m128i a, __m128i b)
{
    __m128i r;
    r.vect_s8 = vorrq_s8(a.vect_s8, b.vect_s8);
    return r;
}

static inline __attribute__((always_inline)) __m128i _mm_xor_si128(__m128i a, __m128i b)
{
    __m128i r;
    r.vect_s8 = veorq_s8(a.vect_s8, b.vect_s8);
    return r;
}

static inline __attribute__((always_inline)) __m128i _mm_andnot_si128(__m128i a, __m128i b)
{
    __m128i r;
    /* _mm_andnot_si128(a, b) = (~a) & b */
    r.vect_s8 = vbicq_s8(b.vect_s8, a.vect_s8);
    return r;
}

#ifdef __cplusplus
}
#endif

/* ========== Conditional Blend & Horizontal Mask ========== */
/* _mm_blendv_epi8 and _mm_movemask_epi8 are handled by avx2ki_noinline.c.
 * That file provides extern "C" function definitions that override the
 * library versions from libavx2neon.so — same C linkage, but our tiny
 * ~3-instruction NEON implementations instead of the library's ~200-instruction
 * byte-by-byte extraction.
 *
 * Why not inline/noinline/macro in this header?
 * - always_inline: causes massive caller code bloat (smithWaterman128_16: 2780→7356B)
 * - noinline function: GCC 10.3.1 ICE on static noinline with __m128i + extern "C"
 * - noinline outside extern "C": conflicts with avx.h's later extern "C" declarations
 * - macro: same bloat as always_inline (inline expansion at every call site)
 * - separate .c file: compiled independently, no caller bloat, no linkage conflicts
 *
 * The Makefile compiles avx2ki_noinline.c and links it before libavx2neon.so,
 * so our definitions take precedence over the library's.

/* ========== Shift (immediate) — specialized for each constant value ========== */
/* _mm_srli_si128 / _mm_slli_si128 use vextq_s8 which needs compile-time const.
 * AVX2KI library provides runtime versions (PLT calls), but bwa-mem2 only uses
 * a small set of immediate values: srli with 1,2,4,8 and slli with 1,2.
 * We provide static inline wrappers for each specific value, so the compiler
 * can embed the constant directly into vextq_s8. */

/* _mm_srli_si128: shift right by imm8 bytes (shift in zeros from left) */
static inline __attribute__((always_inline)) __m128i _mm_srli_si128_1(__m128i a) {
    __m128i r; r.vect_s8 = vextq_s8(a.vect_s8, vdupq_n_s8(0), 1); return r;
}
static inline __attribute__((always_inline)) __m128i _mm_srli_si128_2(__m128i a) {
    __m128i r; r.vect_s8 = vextq_s8(a.vect_s8, vdupq_n_s8(0), 2); return r;
}
static inline __attribute__((always_inline)) __m128i _mm_srli_si128_4(__m128i a) {
    __m128i r; r.vect_s8 = vextq_s8(a.vect_s8, vdupq_n_s8(0), 4); return r;
}
static inline __attribute__((always_inline)) __m128i _mm_srli_si128_8(__m128i a) {
    __m128i r; r.vect_s8 = vextq_s8(a.vect_s8, vdupq_n_s8(0), 8); return r;
}

/* _mm_slli_si128: shift left by imm8 bytes (shift in zeros from right) */
static inline __attribute__((always_inline)) __m128i _mm_slli_si128_1(__m128i a) {
    __m128i r; r.vect_s8 = vextq_s8(vdupq_n_s8(0), a.vect_s8, 15); return r;
}
static inline __attribute__((always_inline)) __m128i _mm_slli_si128_2(__m128i a) {
    __m128i r; r.vect_s8 = vextq_s8(vdupq_n_s8(0), a.vect_s8, 14); return r;
}

/* _mm_extract_epi16: extract 16-bit value at lane imm8 */
static inline __attribute__((always_inline)) int _mm_extract_epi16_0(__m128i a) {
    return (int)vgetq_lane_u16(a.vect_u16, 0);
}

/* ========== Remaining — must use AVX2KI library ========== */

/* ========== Already handled above as noinline wrappers ========== */
/* _mm_blendv_epi8: noinline — prevents caller bloat while eliminating 200-instruction library call */
/* _mm_movemask_epi8: noinline — prevents caller bloat while eliminating library call */

/* NOTE: Macro overrides for _mm_srli/slli/extract are defined in simd_compat.h
 * AFTER avx.h is included, to avoid breaking avx.h's extern "C" declarations. */

#endif /* __ARM_NEON || __aarch64__ */

#endif /* AVX2KI_INLINE_H */
