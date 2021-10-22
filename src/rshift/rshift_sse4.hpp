/*
 *	Optimized SSE4 bit-shifting functions - Copyright (c) 2021 Bertrand LE GAL
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

#ifndef _rshift_sse4_
#define _rshift_sse4_
#ifdef __SSE4_2__

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <immintrin.h>

void permutation_sse4(void* ptr_bit_array, const int32_t nBits, const int32_t nFrames = 1)
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
        const __m128i A0 = _mm_loadu_si128( ((const __m128i*) (ptr_bit_array))     );
        const __m128i A1 = _mm_loadu_si128( ((const __m128i*) (ptr_bit_array)) + 1 );

        const __m128i B0 = _mm_slli_epi64 (A0,  1);
        const __m128i B1 = _mm_srli_epi64 (A0, 63);     // B1

        const __m128i B2 = _mm_slli_epi64 (A1,  1);
        const __m128i B3 = _mm_srli_epi64 (A1, 63);     // B3

        const __m128i C0 = _mm_castpd_si128( _mm_shuffle_pd(_mm_castsi128_pd(B3), _mm_castsi128_pd(B1), 1) );
        const __m128i C1 = _mm_castpd_si128( _mm_shuffle_pd(_mm_castsi128_pd(B1), _mm_castsi128_pd(B3), 1) );

        const __m128i D0 = _mm_or_si128(B0, C0);
        const __m128i D1 = _mm_or_si128(B2, C1);

        _mm_storeu_si128( ((__m128i*)ptr_bit_array),    D0);
        _mm_storeu_si128( ((__m128i*)ptr_bit_array) + 1, D1);
    }
    else if( nBits == 512 )
    {
        const __m128i A0 = _mm_loadu_si128( ((const __m128i*) (ptr_bit_array))     );
        const __m128i A1 = _mm_loadu_si128( ((const __m128i*) (ptr_bit_array)) + 1 );
        const __m128i A2 = _mm_loadu_si128( ((const __m128i*) (ptr_bit_array)) + 2 );
        const __m128i A3 = _mm_loadu_si128( ((const __m128i*) (ptr_bit_array)) + 3 );

        const __m128i B0 = _mm_slli_epi64 (A0,  1);
        const __m128i B2 = _mm_slli_epi64 (A1,  1);
        const __m128i B4 = _mm_slli_epi64 (A2,  1);
        const __m128i B6 = _mm_slli_epi64 (A3,  1);

        const __m128i B1 = _mm_srli_epi64 (A0, 63);
        const __m128i B3 = _mm_srli_epi64 (A1, 63);
        const __m128i B5 = _mm_srli_epi64 (A2, 63);
        const __m128i B7 = _mm_srli_epi64 (A3, 63);

        const __m128i C0 = _mm_castpd_si128( _mm_shuffle_pd(_mm_castsi128_pd(B7), _mm_castsi128_pd(B1), 1) );
        const __m128i C1 = _mm_castpd_si128( _mm_shuffle_pd(_mm_castsi128_pd(B1), _mm_castsi128_pd(B3), 1) );
        const __m128i C2 = _mm_castpd_si128( _mm_shuffle_pd(_mm_castsi128_pd(B3), _mm_castsi128_pd(B5), 1) );
        const __m128i C3 = _mm_castpd_si128( _mm_shuffle_pd(_mm_castsi128_pd(B5), _mm_castsi128_pd(B7), 1) );

        const __m128i D0 = _mm_or_si128(B0, C0);
        const __m128i D1 = _mm_or_si128(B2, C1);
        const __m128i D2 = _mm_or_si128(B4, C2);
        const __m128i D3 = _mm_or_si128(B6, C3);

        _mm_storeu_si128( ((__m128i*)ptr_bit_array),    D0);
        _mm_storeu_si128( ((__m128i*)ptr_bit_array) + 1, D1);
        _mm_storeu_si128( ((__m128i*)ptr_bit_array) + 2, D2);
        _mm_storeu_si128( ((__m128i*)ptr_bit_array) + 3, D3);
    }
    else if( nBits == 1024 )
    {
        const __m128i A0 = _mm_loadu_si128( ((const __m128i*) (ptr_bit_array))     );
        const __m128i A1 = _mm_loadu_si128( ((const __m128i*) (ptr_bit_array)) + 1 );
        const __m128i A2 = _mm_loadu_si128( ((const __m128i*) (ptr_bit_array)) + 2 );
        const __m128i A3 = _mm_loadu_si128( ((const __m128i*) (ptr_bit_array)) + 3 );
        const __m128i A4 = _mm_loadu_si128( ((const __m128i*) (ptr_bit_array)) + 4 );
        const __m128i A5 = _mm_loadu_si128( ((const __m128i*) (ptr_bit_array)) + 5 );
        const __m128i A6 = _mm_loadu_si128( ((const __m128i*) (ptr_bit_array)) + 6 );
        const __m128i A7 = _mm_loadu_si128( ((const __m128i*) (ptr_bit_array)) + 7 );

        const __m128i B0  = _mm_slli_epi64 (A0,  1);
        const __m128i B2  = _mm_slli_epi64 (A1,  1);
        const __m128i B4  = _mm_slli_epi64 (A2,  1);
        const __m128i B6  = _mm_slli_epi64 (A3,  1);
        const __m128i B8  = _mm_slli_epi64 (A4,  1);
        const __m128i B10 = _mm_slli_epi64 (A5,  1);
        const __m128i B12 = _mm_slli_epi64 (A6,  1);
        const __m128i B14 = _mm_slli_epi64 (A7,  1);

        const __m128i B1  = _mm_srli_epi64 (A0, 63);
        const __m128i B3  = _mm_srli_epi64 (A1, 63);
        const __m128i B5  = _mm_srli_epi64 (A2, 63);
        const __m128i B7  = _mm_srli_epi64 (A3, 63);
        const __m128i B9  = _mm_srli_epi64 (A4, 63);
        const __m128i B11 = _mm_srli_epi64 (A5, 63);
        const __m128i B13 = _mm_srli_epi64 (A6, 63);
        const __m128i B15 = _mm_srli_epi64 (A7, 63);

        const __m128i C0 = _mm_castpd_si128( _mm_shuffle_pd(_mm_castsi128_pd(B15), _mm_castsi128_pd(B1),  1) );
        const __m128i C1 = _mm_castpd_si128( _mm_shuffle_pd(_mm_castsi128_pd(B1),  _mm_castsi128_pd(B3),  1) );
        const __m128i C2 = _mm_castpd_si128( _mm_shuffle_pd(_mm_castsi128_pd(B3),  _mm_castsi128_pd(B5),  1) );
        const __m128i C3 = _mm_castpd_si128( _mm_shuffle_pd(_mm_castsi128_pd(B5),  _mm_castsi128_pd(B7),  1) );
        const __m128i C4 = _mm_castpd_si128( _mm_shuffle_pd(_mm_castsi128_pd(B7),  _mm_castsi128_pd(B9),  1) );
        const __m128i C5 = _mm_castpd_si128( _mm_shuffle_pd(_mm_castsi128_pd(B9),  _mm_castsi128_pd(B11), 1) );
        const __m128i C6 = _mm_castpd_si128( _mm_shuffle_pd(_mm_castsi128_pd(B11), _mm_castsi128_pd(B13), 1) );
        const __m128i C7 = _mm_castpd_si128( _mm_shuffle_pd(_mm_castsi128_pd(B13), _mm_castsi128_pd(B15), 1) );

        const __m128i D0 = _mm_or_si128(B0,  C0);
        const __m128i D1 = _mm_or_si128(B2,  C1);
        const __m128i D2 = _mm_or_si128(B4,  C2);
        const __m128i D3 = _mm_or_si128(B6,  C3);
        const __m128i D4 = _mm_or_si128(B8,  C4);
        const __m128i D5 = _mm_or_si128(B10, C5);
        const __m128i D6 = _mm_or_si128(B12, C6);
        const __m128i D7 = _mm_or_si128(B14, C7);

        _mm_storeu_si128( ((__m128i*)ptr_bit_array),    D0);
        _mm_storeu_si128( ((__m128i*)ptr_bit_array) + 1, D1);
        _mm_storeu_si128( ((__m128i*)ptr_bit_array) + 2, D2);
        _mm_storeu_si128( ((__m128i*)ptr_bit_array) + 3, D3);
        _mm_storeu_si128( ((__m128i*)ptr_bit_array) + 4, D4);
        _mm_storeu_si128( ((__m128i*)ptr_bit_array) + 5, D5);
        _mm_storeu_si128( ((__m128i*)ptr_bit_array) + 6, D6);
        _mm_storeu_si128( ((__m128i*)ptr_bit_array) + 7, D7);
    }
    else if( nBits == 2048 )
    {
        const __m128i A0  = _mm_loadu_si128( ((const __m128i*) (ptr_bit_array))        );
        const __m128i A1  = _mm_loadu_si128( ((const __m128i*) (ptr_bit_array)) +  1 );
        const __m128i A2  = _mm_loadu_si128( ((const __m128i*) (ptr_bit_array)) +  2 );
        const __m128i A3  = _mm_loadu_si128( ((const __m128i*) (ptr_bit_array)) +  3 );
        const __m128i A4  = _mm_loadu_si128( ((const __m128i*) (ptr_bit_array)) +  4 );
        const __m128i A5  = _mm_loadu_si128( ((const __m128i*) (ptr_bit_array)) +  5 );
        const __m128i A6  = _mm_loadu_si128( ((const __m128i*) (ptr_bit_array)) +  6 );
        const __m128i A7  = _mm_loadu_si128( ((const __m128i*) (ptr_bit_array)) +  7 );
        const __m128i A8  = _mm_loadu_si128( ((const __m128i*) (ptr_bit_array)) +  8 );
        const __m128i A9  = _mm_loadu_si128( ((const __m128i*) (ptr_bit_array)) +  9 );
        const __m128i A10 = _mm_loadu_si128( ((const __m128i*) (ptr_bit_array)) + 10 );
        const __m128i A11 = _mm_loadu_si128( ((const __m128i*) (ptr_bit_array)) + 11 );
        const __m128i A12 = _mm_loadu_si128( ((const __m128i*) (ptr_bit_array)) + 12 );
        const __m128i A13 = _mm_loadu_si128( ((const __m128i*) (ptr_bit_array)) + 13 );
        const __m128i A14 = _mm_loadu_si128( ((const __m128i*) (ptr_bit_array)) + 14 );
        const __m128i A15 = _mm_loadu_si128( ((const __m128i*) (ptr_bit_array)) + 15 );

        const __m128i B0  = _mm_slli_epi64 (A0,   1);
        const __m128i B2  = _mm_slli_epi64 (A1,   1);
        const __m128i B4  = _mm_slli_epi64 (A2,   1);
        const __m128i B6  = _mm_slli_epi64 (A3,   1);
        const __m128i B8  = _mm_slli_epi64 (A4,   1);
        const __m128i B10 = _mm_slli_epi64 (A5,   1);
        const __m128i B12 = _mm_slli_epi64 (A6,   1);
        const __m128i B14 = _mm_slli_epi64 (A7,   1);
        const __m128i B16 = _mm_slli_epi64 (A8,   1);
        const __m128i B18 = _mm_slli_epi64 (A9,   1);
        const __m128i B20 = _mm_slli_epi64 (A10,  1);
        const __m128i B22 = _mm_slli_epi64 (A11,  1);
        const __m128i B24 = _mm_slli_epi64 (A12,  1);
        const __m128i B26 = _mm_slli_epi64 (A13,  1);
        const __m128i B28 = _mm_slli_epi64 (A14,  1);
        const __m128i B30 = _mm_slli_epi64 (A15,  1);

        const __m128i B1  = _mm_srli_epi64 (A0,  63);
        const __m128i B3  = _mm_srli_epi64 (A1,  63);
        const __m128i B5  = _mm_srli_epi64 (A2,  63);
        const __m128i B7  = _mm_srli_epi64 (A3,  63);
        const __m128i B9  = _mm_srli_epi64 (A4,  63);
        const __m128i B11 = _mm_srli_epi64 (A5,  63);
        const __m128i B13 = _mm_srli_epi64 (A6,  63);
        const __m128i B15 = _mm_srli_epi64 (A7,  63);
        const __m128i B17 = _mm_srli_epi64 (A8,  63);
        const __m128i B19 = _mm_srli_epi64 (A9,  63);
        const __m128i B21 = _mm_srli_epi64 (A10, 63);
        const __m128i B23 = _mm_srli_epi64 (A11, 63);
        const __m128i B25 = _mm_srli_epi64 (A12, 63);
        const __m128i B27 = _mm_srli_epi64 (A13, 63);
        const __m128i B29 = _mm_srli_epi64 (A14, 63);
        const __m128i B31 = _mm_srli_epi64 (A15, 63);

        const __m128i C0  = _mm_castpd_si128( _mm_shuffle_pd(_mm_castsi128_pd(B31), _mm_castsi128_pd(B1),  1) );
        const __m128i C1  = _mm_castpd_si128( _mm_shuffle_pd(_mm_castsi128_pd(B1),  _mm_castsi128_pd(B3),  1) );
        const __m128i C2  = _mm_castpd_si128( _mm_shuffle_pd(_mm_castsi128_pd(B3),  _mm_castsi128_pd(B5),  1) );
        const __m128i C3  = _mm_castpd_si128( _mm_shuffle_pd(_mm_castsi128_pd(B5),  _mm_castsi128_pd(B7),  1) );
        const __m128i C4  = _mm_castpd_si128( _mm_shuffle_pd(_mm_castsi128_pd(B7),  _mm_castsi128_pd(B9),  1) );
        const __m128i C5  = _mm_castpd_si128( _mm_shuffle_pd(_mm_castsi128_pd(B9),  _mm_castsi128_pd(B11), 1) );
        const __m128i C6  = _mm_castpd_si128( _mm_shuffle_pd(_mm_castsi128_pd(B11), _mm_castsi128_pd(B13), 1) );
        const __m128i C7  = _mm_castpd_si128( _mm_shuffle_pd(_mm_castsi128_pd(B13), _mm_castsi128_pd(B15), 1) );
        const __m128i C8  = _mm_castpd_si128( _mm_shuffle_pd(_mm_castsi128_pd(B15), _mm_castsi128_pd(B17), 1) );
        const __m128i C9  = _mm_castpd_si128( _mm_shuffle_pd(_mm_castsi128_pd(B17), _mm_castsi128_pd(B19), 1) );
        const __m128i C10 = _mm_castpd_si128( _mm_shuffle_pd(_mm_castsi128_pd(B19), _mm_castsi128_pd(B21), 1) );
        const __m128i C11 = _mm_castpd_si128( _mm_shuffle_pd(_mm_castsi128_pd(B21), _mm_castsi128_pd(B23), 1) );
        const __m128i C12 = _mm_castpd_si128( _mm_shuffle_pd(_mm_castsi128_pd(B23), _mm_castsi128_pd(B25), 1) );
        const __m128i C13 = _mm_castpd_si128( _mm_shuffle_pd(_mm_castsi128_pd(B25), _mm_castsi128_pd(B27), 1) );
        const __m128i C14 = _mm_castpd_si128( _mm_shuffle_pd(_mm_castsi128_pd(B27), _mm_castsi128_pd(B29), 1) );
        const __m128i C15 = _mm_castpd_si128( _mm_shuffle_pd(_mm_castsi128_pd(B29), _mm_castsi128_pd(B31), 1) );

        const __m128i D0  = _mm_or_si128(B0,  C0);
        const __m128i D1  = _mm_or_si128(B2,  C1);
        const __m128i D2  = _mm_or_si128(B4,  C2);
        const __m128i D3  = _mm_or_si128(B6,  C3);
        const __m128i D4  = _mm_or_si128(B8,  C4);
        const __m128i D5  = _mm_or_si128(B10, C5);
        const __m128i D6  = _mm_or_si128(B12, C6);
        const __m128i D7  = _mm_or_si128(B14, C7);
        const __m128i D8  = _mm_or_si128(B16, C8);
        const __m128i D9  = _mm_or_si128(B18, C9);
        const __m128i D10 = _mm_or_si128(B20, C10);
        const __m128i D11 = _mm_or_si128(B22, C11);
        const __m128i D12 = _mm_or_si128(B24, C12);
        const __m128i D13 = _mm_or_si128(B26, C13);
        const __m128i D14 = _mm_or_si128(B28, C14);
        const __m128i D15 = _mm_or_si128(B30, C15);

        _mm_storeu_si128( ((__m128i*)ptr_bit_array),    D0);
        _mm_storeu_si128( ((__m128i*)ptr_bit_array) +  1, D1);
        _mm_storeu_si128( ((__m128i*)ptr_bit_array) +  2, D2);
        _mm_storeu_si128( ((__m128i*)ptr_bit_array) +  3, D3);
        _mm_storeu_si128( ((__m128i*)ptr_bit_array) +  4, D4);
        _mm_storeu_si128( ((__m128i*)ptr_bit_array) +  5, D5);
        _mm_storeu_si128( ((__m128i*)ptr_bit_array) +  6, D6);
        _mm_storeu_si128( ((__m128i*)ptr_bit_array) +  7, D7);
        _mm_storeu_si128( ((__m128i*)ptr_bit_array) +  8, D8);
        _mm_storeu_si128( ((__m128i*)ptr_bit_array) +  9, D9);
        _mm_storeu_si128( ((__m128i*)ptr_bit_array) + 10, D10);
        _mm_storeu_si128( ((__m128i*)ptr_bit_array) + 11, D11);
        _mm_storeu_si128( ((__m128i*)ptr_bit_array) + 12, D12);
        _mm_storeu_si128( ((__m128i*)ptr_bit_array) + 13, D13);
        _mm_storeu_si128( ((__m128i*)ptr_bit_array) + 14, D14);
        _mm_storeu_si128( ((__m128i*)ptr_bit_array) + 15, D15);
    }
    else
    {
        printf("permutation_avx2(%d) : AVX2 IMPLEMENTATION NOT DONE YET !\n", nBits);
        exit( EXIT_FAILURE );
    }

}

#endif
#endif