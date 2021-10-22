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

#ifndef _rshift_x86_
#define _rshift_x86_

#include <cstdint>

void permutation_x86(void* ptr_bit_array, const int32_t nBits, const int32_t nFrames = 1)
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
        uint64_t* bit_array = (uint64_t*)ptr_bit_array;
        const uint64_t outbit_1  = bit_array[0];
        const uint64_t outbit_2  = bit_array[1];
        bit_array[0] = (bit_array[0] << 1) | (outbit_2 >> 63);
        bit_array[1] = (bit_array[1] << 1) | (outbit_1 >> 63);
    }
    else if( nBits == 256 )
    {
        uint64_t* bit_array = (uint64_t*)ptr_bit_array;
        const uint64_t outbit_1  = bit_array[0];
        const uint64_t outbit_2  = bit_array[1];
        const uint64_t outbit_3  = bit_array[2];
        const uint64_t outbit_4  = bit_array[3];
        bit_array[0] = (bit_array[0] << 1) | (outbit_4 >> 63);
        bit_array[1] = (bit_array[1] << 1) | (outbit_1 >> 63);
        bit_array[2] = (bit_array[2] << 1) | (outbit_2 >> 63);
        bit_array[3] = (bit_array[3] << 1) | (outbit_3 >> 63);
    }
    else if( nBits == 512 )
    {
        constexpr int32_t bytes = 8;
        uint64_t* bit_array = (uint64_t*)ptr_bit_array;
        uint64_t tmp[bytes];

        #pragma unroll
        for(int32_t x = 0; x < bytes; x += 1)
            tmp[x] = bit_array[x];
        bit_array[0] = (tmp[0] << 1) | (tmp[bytes-1] >> 63);

        #pragma unroll
        for(int32_t x = 1; x < bytes; x += 1)
            bit_array[x] = (tmp[x] << 1) | (tmp[x-1] >> 63);
    }
    else if( nBits == 1024 )
    {
        constexpr int32_t bytes = 16;
        uint64_t* bit_array = (uint64_t*)ptr_bit_array;
        uint64_t tmp[bytes];

        #pragma unroll
        for(int32_t x = 0; x < bytes; x += 1)
            tmp[x] = bit_array[x];
        bit_array[0] = (tmp[0] << 1) | (tmp[bytes-1] >> 63);

        #pragma unroll
        for(int32_t x = 1; x < bytes; x += 1)
            bit_array[x] = (tmp[x] << 1) | (tmp[x-1] >> 63);
    }
    else if( nBits == 2048 )
    {
        constexpr int32_t bytes = 32;
        uint64_t* bit_array = (uint64_t*)ptr_bit_array;
        uint64_t tmp[bytes];

        #pragma unroll
        for(int32_t x = 0; x < bytes; x += 1)
            tmp[x] = bit_array[x];
        bit_array[0] = (tmp[0] << 1) | (tmp[bytes-1] >> 63);

        #pragma unroll
        for(int32_t x = 1; x < bytes; x += 1)
            bit_array[x] = (tmp[x] << 1) | (tmp[x-1] >> 63);
    }
}

#endif