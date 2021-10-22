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

//#ifndef __ARM_NEON__
//    #define __ARM_NEON__ // Mode emulation
//#endif

#include "./rshift/rshift_x86.hpp"
#include "./rshift/rshift_sse4.hpp"
#include "./rshift/rshift_avx2.hpp"

#include "./bit_pack/x86/bit_pack_x86.hpp"
#include "./bit_unpack/x86/bit_unpack_x86.hpp"

#include <cstring>
#include <chrono>

void dump_uint8_bits(const uint8_t* bits, const int32_t ll)
{
    for(int32_t i = 0; i < ll; i += 1)
    {
             if( (i%128 == 0) && (i!=0) ) printf("\n");
        //else if( (i%8   == 0) && (i!=0) ) printf(" ");
        printf("%d", bits[i]);
    }
    printf("\n");
}

void dump_bits(const uint8_t* bits, const int32_t ll)
{
    for(int32_t i = 0; i < ll; i += 8)
    {
//        if( (i%128 == 0) && (i!=0) ) printf("\n");
        const uint32_t v = bits[i/8];
        for(int32_t q = 0; q < 8; q += 1)
            printf("%d", ((v >> q) & 0x01));
    }
    printf("\n");
}


void p128_hex_u64(__m128i in) {
    alignas(16) unsigned long long v[2];  // uint64_t might give format-string warnings with %llx; it's just long in some ABIs
    _mm_store_si128((__m128i*)v, in);
    printf("v2_u64: %llx %llx\n", v[0], v[1]);
}



bool check_result(const uint8_t* i_bits, const uint8_t* t_bits, const int32_t size_bits)
{
    const int32_t size_bytes = size_bits / 8;
    for(int32_t x = 0; x < size_bytes; x += 1)
    {
        if( i_bits[x] != t_bits[x] )
            return false;
    }
    return true;
}



int main(int argc, char* argv[])
{

#if defined (__APPLE__)
    printf("(II) Benchmarking the bit_pack/bit_unpack functions on MacOS\n");
#elif defined (__linux__)
    printf("(II) Benchmarking the bit_pack/bit_unpack functions on Linux\n");
#else
    printf("(II) Benchmarking the bit_pack/bit_unpack functions on a undefined platform\n");
#endif


#if defined (__clang__)
    printf("(II) Code compiled with LLVM (%d.%d.%d)\n", __clang_major__, __clang_minor__, __clang_patchlevel__);
#elif defined (__GNUC__)
    printf("(II) Code compiled with GCC (%d.%d.%d)\n", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#else
    printf("(II) Code compiled with UNKWON compiler\n");
#endif

    const int32_t v_begin =   32;
    const int32_t v_end   = 2048;
    const int32_t v_step  =   2;

    printf("|  LENGTH  |    x86   |   SSE4   |   AVX2   |\n");

    for( int32_t size_bits = v_begin; size_bits <= v_end; size_bits *= v_step )
    {
        const int32_t bench_loop = 16*67108864 / size_bits;

        const int32_t size_bytes = size_bits / 8;
        printf("| %8d |", size_bits);

        uint8_t i_bits   [size_bits ];
        uint8_t x86_bits [size_bytes];   // reference bit array from x86 code
        uint8_t sse4_bits[size_bytes];   // optimized bit array from x86 code
        uint8_t avx2_bits[size_bytes];   // optimized bit array from x86 code

        for(int i = 0; i < size_bits; i+= 1)
        {
            i_bits[i] = (i == 0); //rand()%2;
        }

        //
        // On compresse les bytes en bits
        //

        bit_pack_x86(x86_bits,  i_bits, size_bits);
        bit_pack_x86(sse4_bits, i_bits, size_bits);
        bit_pack_x86(avx2_bits, i_bits, size_bits);

#if 0
        for(int32_t i = 0; i < size_bits; i+= 1)
        {
            printf("%3d : ", i); dump_bits(avx2_bits, size_bits);
            permutation_avx2(avx2_bits, size_bits);
        }
        dump_bits(avx2_bits, size_bits);
//        exit( EXIT_FAILURE );
#endif

            auto start = std::chrono::steady_clock::now();
            for(int32_t z = 0; z < bench_loop; z += 1)
                for(int32_t i = 0; i < size_bits; i+= 1)
                    permutation_x86 (x86_bits, size_bits);
            auto end = std::chrono::steady_clock::now();
            const int32_t time_x86 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / bench_loop;
            printf("  %6d  |", time_x86);

            //
            ////////////////////////////////////////////////////
            //
#ifdef __SSE4_2__
            auto start_sse4 = std::chrono::steady_clock::now();
            for(int32_t z = 0; z < bench_loop; z += 1)
                for(int32_t i = 0; i < size_bits; i+= 1)
                    permutation_sse4(sse4_bits, size_bits);
            auto end_sse4 = std::chrono::steady_clock::now();
            const int32_t time_sse4 = std::chrono::duration_cast<std::chrono::nanoseconds>(end_sse4 - start_sse4).count() / bench_loop;
            if( check_result(x86_bits, sse4_bits, size_bits) == false )
                printf("  \x1B[31m%6d\x1B[0m  |", time_sse4);
            else
                printf("  \x1B[32m%6d\x1B[0m  |", time_sse4);
#endif
            //
            ////////////////////////////////////////////////////
            //
#ifdef __AVX2__
            auto start_avx2 = std::chrono::steady_clock::now();
            for(int32_t z = 0; z < bench_loop; z += 1)
                for(int32_t i = 0; i < size_bits; i+= 1)
                    permutation_avx2(avx2_bits, size_bits);
            auto end_avx2 = std::chrono::steady_clock::now();
            const int32_t time_avx2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end_avx2 - start_avx2).count() / bench_loop;
            if( check_result(x86_bits, avx2_bits, size_bits) == false )
                printf("  \x1B[31m%6d\x1B[0m  |", time_avx2);
            else
                printf("  \x1B[32m%6d\x1B[0m  |", time_avx2);
#endif
            //
            ////////////////////////////////////////////////////
            //
            printf("\n");
    }



    return EXIT_SUCCESS;
}
    
