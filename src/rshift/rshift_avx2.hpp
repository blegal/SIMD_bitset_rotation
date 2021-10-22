/*
*	Optimized bit-packing and bit-unpacking functions - Copyright (c) 2021 Bertrand LE GAL
 *
 *  This software is provided 'as-is', without any express or
 *  implied warranty. In no event will the authors be held
 *  liable for any damages arising from the use of this software.
 *
 *  Permission is granted to anyone to use this software for any purpose,
 *  including commercial applications, and to alter it and redistribute
 *  it freely, subject to the following restrictions:
 *
 *  1. The origin of this software must not be misrepresented;
 *  you must not claim that you wrote the original software.
 *  If you use this software in a product, an acknowledgment
 *  in the product documentation would be appreciated but
 *  is not required.
 *
 *  2. Altered source versions must be plainly marked as such,
 *  and must not be misrepresented as being the original software.
 *
 *  3. This notice may not be removed or altered from any
 *  source distribution.
 *
 */

#ifndef _rshift_avx2_
#define _rshift_avx2_
#ifdef __AVX2__

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <immintrin.h>

void permutation_avx2(void* ptr_bit_array, const int32_t nBits, const int32_t nFrames = 1)
{
    if( nBits == 32 )
    {
        uint32_t* bit_array = (uint32_t*)ptr_bit_array;
        bit_array[0] = (bit_array[0] << 1) | (bit_array[0] >> 31);
    }
    else if( nBits == 64 )
    {
        uint64_t* bit_array = (uint64_t*)ptr_bit_array;
        bit_array[0] = (bit_array[0] << 1) | (bit_array[0] >> 63);
    }
    else if( nBits == 128 )
    {
        const __m128i A  = _mm_loadu_si128((const __m128i*) ptr_bit_array);
        const __m128i B0 = _mm_slli_epi64 (A,  1);
        const __m128i B1 = _mm_srli_epi64 (A, 63);
        const __m128i C0 = _mm_castpd_si128( _mm_permute_pd(_mm_castsi128_pd(B1), 1) );
        const __m128i D0 = _mm_or_si128(B0, C0);
        _mm_storeu_si128((__m128i*) ptr_bit_array, D0);
    }
    else if( nBits == 256 )
    {
        const __m256i A  = _mm256_loadu_si256((const __m256i*) ptr_bit_array);
        const __m256i B0 = _mm256_slli_epi64 (A,  1);
        const __m256i B1 = _mm256_srli_epi64 (A, 63);
        const __m256i C0 = _mm256_permute4x64_epi64(B1, 0x93);
        const __m256i D0 = _mm256_or_si256(B0, C0);
        _mm256_storeu_si256((__m256i*) ptr_bit_array, D0);
    }
    else if( nBits == 512 )
    {
        const __m256i A0 = _mm256_loadu_si256( ((const __m256i*)ptr_bit_array) + 0 );
        const __m256i A1 = _mm256_loadu_si256( ((const __m256i*)ptr_bit_array) + 1);

        const __m256i B0 = _mm256_slli_epi64 (A0,  1);
        const __m256i B1 = _mm256_slli_epi64 (A1,  1);

        const __m256i C0 = _mm256_srli_epi64 (A0, 63);
        const __m256i C1 = _mm256_srli_epi64 (A1, 63);

        const __m256d D0 = _mm256_permute4x64_pd (_mm256_castsi256_pd(C0), 0x93);
        const __m256d D1 = _mm256_permute4x64_pd (_mm256_castsi256_pd(C1), 0x93);

        const __m256i E0 = _mm256_castpd_si256( _mm256_blend_pd (D0, D1, 0x01) );
        const __m256i E1 = _mm256_castpd_si256( _mm256_blend_pd (D1, D0, 0x01) );

        const __m256i F0 = _mm256_or_si256(B0, E0);
        const __m256i F1 = _mm256_or_si256(B1, E1);

        _mm256_storeu_si256( ((__m256i*)ptr_bit_array) + 0, F0);
        _mm256_storeu_si256( ((__m256i*)ptr_bit_array) + 1, F1);
    }
    else if( nBits == 1024 )
    {
        const __m256i A0 = _mm256_loadu_si256( ((const __m256i*)ptr_bit_array) + 0 );
        const __m256i A1 = _mm256_loadu_si256( ((const __m256i*)ptr_bit_array) + 1);
        const __m256i A2 = _mm256_loadu_si256( ((const __m256i*)ptr_bit_array) + 2);
        const __m256i A3 = _mm256_loadu_si256( ((const __m256i*)ptr_bit_array) + 3);

        const __m256i B0 = _mm256_slli_epi64 (A0,  1);
        const __m256i B1 = _mm256_slli_epi64 (A1,  1);
        const __m256i B2 = _mm256_slli_epi64 (A2,  1);
        const __m256i B3 = _mm256_slli_epi64 (A3,  1);

        const __m256i C0 = _mm256_srli_epi64 (A0, 63);
        const __m256i C1 = _mm256_srli_epi64 (A1, 63);
        const __m256i C2 = _mm256_srli_epi64 (A2, 63);
        const __m256i C3 = _mm256_srli_epi64 (A3, 63);

        const __m256d D0 = _mm256_permute4x64_pd (_mm256_castsi256_pd(C0), 0x93);
        const __m256d D1 = _mm256_permute4x64_pd (_mm256_castsi256_pd(C1), 0x93);
        const __m256d D2 = _mm256_permute4x64_pd (_mm256_castsi256_pd(C2), 0x93);
        const __m256d D3 = _mm256_permute4x64_pd (_mm256_castsi256_pd(C3), 0x93);

        const __m256i E0 = _mm256_castpd_si256( _mm256_blend_pd (D0, D3, 0x01) );
        const __m256i E1 = _mm256_castpd_si256( _mm256_blend_pd (D1, D0, 0x01) );
        const __m256i E2 = _mm256_castpd_si256( _mm256_blend_pd (D2, D1, 0x01) );
        const __m256i E3 = _mm256_castpd_si256( _mm256_blend_pd (D3, D2, 0x01) );

        const __m256i F0 = _mm256_or_si256(B0, E0);
        const __m256i F1 = _mm256_or_si256(B1, E1);
        const __m256i F2 = _mm256_or_si256(B2, E2);
        const __m256i F3 = _mm256_or_si256(B3, E3);

        _mm256_storeu_si256( ((__m256i*)ptr_bit_array) + 0, F0);
        _mm256_storeu_si256( ((__m256i*)ptr_bit_array) + 1, F1);
        _mm256_storeu_si256( ((__m256i*)ptr_bit_array) + 2, F2);
        _mm256_storeu_si256( ((__m256i*)ptr_bit_array) + 3, F3);
    }
    else if( nBits == 2048 )
    {
        const __m256i A0 = _mm256_loadu_si256( ((const __m256i*)ptr_bit_array) + 0);
        const __m256i A1 = _mm256_loadu_si256( ((const __m256i*)ptr_bit_array) + 1);
        const __m256i A2 = _mm256_loadu_si256( ((const __m256i*)ptr_bit_array) + 2);
        const __m256i A3 = _mm256_loadu_si256( ((const __m256i*)ptr_bit_array) + 3);
        const __m256i A4 = _mm256_loadu_si256( ((const __m256i*)ptr_bit_array) + 4);
        const __m256i A5 = _mm256_loadu_si256( ((const __m256i*)ptr_bit_array) + 5);
        const __m256i A6 = _mm256_loadu_si256( ((const __m256i*)ptr_bit_array) + 6);
        const __m256i A7 = _mm256_loadu_si256( ((const __m256i*)ptr_bit_array) + 7);

        const __m256i B0 = _mm256_slli_epi64 (A0,  1);
        const __m256i B1 = _mm256_slli_epi64 (A1,  1);
        const __m256i B2 = _mm256_slli_epi64 (A2,  1);
        const __m256i B3 = _mm256_slli_epi64 (A3,  1);
        const __m256i B4 = _mm256_slli_epi64 (A4,  1);
        const __m256i B5 = _mm256_slli_epi64 (A5,  1);
        const __m256i B6 = _mm256_slli_epi64 (A6,  1);
        const __m256i B7 = _mm256_slli_epi64 (A7,  1);

        const __m256i C0 = _mm256_srli_epi64 (A0, 63);
        const __m256i C1 = _mm256_srli_epi64 (A1, 63);
        const __m256i C2 = _mm256_srli_epi64 (A2, 63);
        const __m256i C3 = _mm256_srli_epi64 (A3, 63);
        const __m256i C4 = _mm256_srli_epi64 (A4, 63);
        const __m256i C5 = _mm256_srli_epi64 (A5, 63);
        const __m256i C6 = _mm256_srli_epi64 (A6, 63);
        const __m256i C7 = _mm256_srli_epi64 (A7, 63);

        const __m256d D0 = _mm256_permute4x64_pd (_mm256_castsi256_pd(C0), 0x93);
        const __m256d D1 = _mm256_permute4x64_pd (_mm256_castsi256_pd(C1), 0x93);
        const __m256d D2 = _mm256_permute4x64_pd (_mm256_castsi256_pd(C2), 0x93);
        const __m256d D3 = _mm256_permute4x64_pd (_mm256_castsi256_pd(C3), 0x93);
        const __m256d D4 = _mm256_permute4x64_pd (_mm256_castsi256_pd(C4), 0x93);
        const __m256d D5 = _mm256_permute4x64_pd (_mm256_castsi256_pd(C5), 0x93);
        const __m256d D6 = _mm256_permute4x64_pd (_mm256_castsi256_pd(C6), 0x93);
        const __m256d D7 = _mm256_permute4x64_pd (_mm256_castsi256_pd(C7), 0x93);

        const __m256i E0 = _mm256_castpd_si256( _mm256_blend_pd (D0, D7, 0x01) );
        const __m256i E1 = _mm256_castpd_si256( _mm256_blend_pd (D1, D0, 0x01) );
        const __m256i E2 = _mm256_castpd_si256( _mm256_blend_pd (D2, D1, 0x01) );
        const __m256i E3 = _mm256_castpd_si256( _mm256_blend_pd (D3, D2, 0x01) );
        const __m256i E4 = _mm256_castpd_si256( _mm256_blend_pd (D4, D3, 0x01) );
        const __m256i E5 = _mm256_castpd_si256( _mm256_blend_pd (D5, D4, 0x01) );
        const __m256i E6 = _mm256_castpd_si256( _mm256_blend_pd (D6, D5, 0x01) );
        const __m256i E7 = _mm256_castpd_si256( _mm256_blend_pd (D7, D6, 0x01) );

        const __m256i F0 = _mm256_or_si256(B0, E0);
        const __m256i F1 = _mm256_or_si256(B1, E1);
        const __m256i F2 = _mm256_or_si256(B2, E2);
        const __m256i F3 = _mm256_or_si256(B3, E3);
        const __m256i F4 = _mm256_or_si256(B4, E4);
        const __m256i F5 = _mm256_or_si256(B5, E5);
        const __m256i F6 = _mm256_or_si256(B6, E6);
        const __m256i F7 = _mm256_or_si256(B7, E7);

        _mm256_storeu_si256( ((__m256i*)ptr_bit_array) + 0, F0);
        _mm256_storeu_si256( ((__m256i*)ptr_bit_array) + 1, F1);
        _mm256_storeu_si256( ((__m256i*)ptr_bit_array) + 2, F2);
        _mm256_storeu_si256( ((__m256i*)ptr_bit_array) + 3, F3);
        _mm256_storeu_si256( ((__m256i*)ptr_bit_array) + 4, F4);
        _mm256_storeu_si256( ((__m256i*)ptr_bit_array) + 5, F5);
        _mm256_storeu_si256( ((__m256i*)ptr_bit_array) + 6, F6);
        _mm256_storeu_si256( ((__m256i*)ptr_bit_array) + 7, F7);
    }
    else
    {
        printf("permutation_avx2(%d) : AVX2 IMPLEMENTATION NOT DONE YET !\n", nBits);
        exit( EXIT_FAILURE );
    }
}

#endif
#endif