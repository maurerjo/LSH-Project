/*
 * lsh.c
 *
 *  Created on: May 4, 2017
 *      Author: jonathan
 */

#include "immintrin.h"
#include "lsh.h"
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#define _mm256_set_m128(va, vb) \
        _mm256_insertf128_ps(_mm256_castps128_ps256(vb), va, 1)

// *Really* minimal PCG32 code / (c) 2014 M.E. O'Neill / pcg-random.org
// Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)
typedef struct { uint64_t state;  uint64_t inc; } pcg32_random_t;

pcg32_random_t rng;

uint32_t pcg32_random_r(pcg32_random_t* rng) {
    uint64_t oldstate = rng->state;
    // Advance internal state
    rng->state = oldstate * 6364136223846793005ULL + (rng->inc|1);
    // Calculate output function (XSH RR), uses old state for max ILP
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

//Returns +1 or -1
int rngi() {
  return 1 - 2 * (pcg32_random_r(&rng) & 0x1);
}

void init_rng() {
  rng.state = 42;
  rng.inc = 1337;
}

const int kUndefined = -1;

//for faster look-up
float * HMatVecC;
int HMatVecLen = 0;

float * RotMat;

float * Data;
int num_points = 0;
int num_dimensions = 0;

int * tables;
int num_tables = 0;
int table_size = 0;

float * rotation_vecs;
int num_rotations = 0;
int k = 0;

void SetData(float* data_pointer, int points, int dimensions) {
  Data = data_pointer;
  num_points = points;
  num_dimensions = dimensions;
}

void SetTables(int num_tables_, int table_size_) {
  num_tables = num_tables_;
  table_size = table_size_;
  tables = (int *)malloc(num_tables*table_size*sizeof(int));
  for (int idx = 0; idx < num_tables * table_size; idx++) {
    tables[idx] = kUndefined;
  }
}

void SetRotationVecs(int num_tables_, int num_rotations_, int k_, int num_dimensions_) {
  num_rotations = num_rotations_;
  k = k_;
  int num_vals = num_tables_*num_rotations_*k_*num_dimensions_;
  rotation_vecs = (float *)malloc(num_vals*sizeof(float));
  for (int i = 0; i < num_vals; i++) {
    rotation_vecs[i] = rngi();
  }
}

void set_rotation_vec_entry(int table_idx, int hash_rotation_idx, int rotation_idx, int dim_idx, float value) {
  rotation_vecs[table_idx * k * num_rotations * num_dimensions
                + hash_rotation_idx * num_rotations * num_dimensions
                + rotation_idx * num_dimensions
                + dim_idx] = value;
}

void SetHMatVecC(int dim) {
  int log_dim = (int)floor(log2(dim));
  int h_dim = 1<<log_dim;
  HMatVecC = (float *)malloc(h_dim*sizeof(float));
  HMatVecLen = h_dim;
  //hadamard scalar
  float scalar = pow(2,-(log_dim/2.0));
  for(int i = 0; i<h_dim; i++){
    HMatVecC[i] = scalar * (1 - ((_mm_popcnt_u32(i) & 0x1) << 1));
  }
}


//sets HMatC to a standard hadamard matrix and precomputes all rotations
void precomputeRotation(){
    //precompute rotations
    int log_dim = (int)floor(log2(num_dimensions));
    int h_dim = 1<<log_dim;
    if (h_dim != HMatVecLen) {
        SetHMatVecC(num_dimensions);
    }
    //allocate memory
    RotMat = (float *) malloc(num_tables * k * HMatVecLen *HMatVecLen* sizeof(float));
    float tempRot[HMatVecLen*HMatVecLen];
    for(int table_idx = 0; table_idx<num_tables;table_idx++){
        for(int hash_rotation_idx = 0; hash_rotation_idx<k;hash_rotation_idx++){
            float * currentRot = &RotMat[(table_idx*k+hash_rotation_idx)*HMatVecLen*HMatVecLen];
            //initialize to 0
            for(int i = 0; i < HMatVecLen*HMatVecLen;i++){
                currentRot[i]=0;
            }
            //currentRot = I
            for(int i = 0; i < HMatVecLen;i++){
                currentRot[i*HMatVecLen+i]=1;
            }

            //initialize tempRot to be currentRot, needed if we want to inverse the order of MMM
            for(int i = 0; i < HMatVecLen*HMatVecLen;i++){
                tempRot[i] = currentRot[i];
            }

            for(int rotation_idx = 0; rotation_idx<num_rotations;rotation_idx++){


                //MMM with hadamard
                //currentRot *= H
                for(int i = 0; i<HMatVecLen;i++){
                    for(int ii = 0; ii<HMatVecLen;ii++) {
                        float temp = 0;
                        for (int i3 = 0; i3 < HMatVecLen; i3++) {
                            temp += tempRot[i * HMatVecLen + i3] *
                                    HMatVecC[ii&i3] *
                                    rotation_vecs[table_idx * k * num_rotations * num_dimensions
                                                  + hash_rotation_idx * num_rotations * num_dimensions
                                                  + rotation_idx * num_dimensions + ii];
                        }
                        currentRot[i * HMatVecLen + ii] = temp;
                    }
                }//end hadamard mmm
                for(int i = 0; i < HMatVecLen*HMatVecLen;i++){
                    tempRot[i] = currentRot[i];
                }
/*
                //multiplication with random +/-1 diag matrix
                // currentRot *= diag(r)
                for(int i = 0; i<HMatVecLen;i++){
                    for(int ii = 0; ii<HMatVecLen;ii++){
                        currentRot[i*HMatVecLen+ii]*=;
                        tempRot[i*HMatVecLen+ii] = currentRot[i*HMatVecLen+ii];
                    }
                }//end random diag*/
            }
            //transpose for better access pattern in avx
            for(int i = 0; i<HMatVecLen;i++) {
                for (int ii = 0; ii < HMatVecLen; ii++) {
                    currentRot[i*HMatVecLen+ii]=tempRot[ii*HMatVecLen+i];
                }
            }
        }
    }
}

void set_table_entry(int table_idx, unsigned int hash, int entry_idx) {
  tables[table_idx * table_size + (hash%table_size)] = entry_idx;
}

int get_neighbor(int table_idx, unsigned int hash) {
  return tables[table_idx * table_size + (hash%table_size)];
}


//load time: load 4 * dim bytes = dim/2 cycle
//runtime: 12 * dim cycle
//flops: 3 * dim (only counting compares and max and sub, since the other are integer operations)
//Performance: 3/8 flops/cycle
int locality_sensitive_hash(float *data, int dim) {
    int res = 0;
    //for(int i = 0;i<20;i++){
    float best = data[0];
    if (-data[0] > best) {//latency 4
        best = -data[0];
        res = dim;
    }
    for (int ii = 1; ii < dim; ++ii) {
        if (data[ii] > best) {//latency 4 (part of the path because of if else
            best = data[ii];
            res = ii;
        } else if (-data[ii] > best) {//latency 4+4
            best = -data[ii];
            res = ii + dim;//latency 1
        }
    }//}
    return res;
}


//load time: load 4 * dim bytes = dim/2 cycle
//runtime: 12 / 8 * dim cycle
//flops: 5 * dim (only counting compares and max and sub, since the other are integer operations)
//Notice we do more flops, but improve ilp by that and decrease runtime 8-fold using avx
//Performance: 5/1.5 flops/cycle
int locality_sensitive_hash_optimized(float *data, int dim) {
    int res = 0;
    //float best = data[0];
    __m256 best = _mm256_loadu_ps(data);
    __m256 ZERO = _mm256_setzero_ps();
    __m256 best_neg = _mm256_sub_ps(ZERO,best);//0.5 cycle
    __m256i index = _mm256_set_epi32(0,1,2,3,4,5,6,7);
    __m256i iter = _mm256_set1_epi32(8);
    __m256i allONE = _mm256_set1_epi32(-1);
    __m256i best_idx = index;
    __m256i best_idx_neg = index;
    for (int ii = 8; ii < dim; ii+=8) {//negativ and positive for maximal ilp
        index = _mm256_add_epi32(index,iter);
        __m256 current = _mm256_loadu_ps(data+ii);
        __m256 current_neg = _mm256_sub_ps(ZERO,current);//0.5 cycle cpu occupied, 4 latency
        __m256 compare = _mm256_cmp_ps(best, current, 1);//0.5 cycle cpu occupied
        __m256i compare_i = _mm256_castps_si256(compare);
        best_idx = _mm256_or_si256(best_idx,compare_i);//set all changed indexes to 0xFFFFFFFF, 0.333 cycle cpu occupied
        __m256i xor_factor = _mm256_xor_si256(compare_i,allONE);//0.333 cycle cpu occupied
        __m256i and_factor = _mm256_or_si256(xor_factor,index);//0.333 cycle cpu occupied
        best_idx = _mm256_and_si256(best_idx,and_factor);//set new best indexes, 0.333 cycle cpu occupied
        best = _mm256_max_ps(best,current);//set new best values, 0.5 cycles cpu occupied

        __m256 compare_neg = _mm256_cmp_ps(best_neg, current_neg, 1);//4 latency
        __m256i compare_neg_i = _mm256_castps_si256(compare_neg);//0 latency
        best_idx_neg = _mm256_or_si256(best_idx_neg,compare_neg_i);//set all changed indexes to 0xFFFFFFFF, 1 latency
        __m256i xor_factor_neg = _mm256_xor_si256(compare_neg_i,allONE);//1 latency
        __m256i and_factor_neg = _mm256_or_si256(xor_factor_neg,index);//1 latency
        best_idx_neg = _mm256_and_si256(best_idx_neg,and_factor_neg);//set new best indexes, 1 latency
        best_neg = _mm256_max_ps(best_neg,current_neg);//set new best negative values, 4 latency (not part of longest chain)
    }//~8 cycle worth of operations, longest latency chain is 12 cycle
    __m256 compare = _mm256_cmp_ps(best, best_neg, 1);//4 latency
    __m256i compare_i = _mm256_castps_si256(compare);//0 latency
    best_idx = _mm256_or_si256(best_idx,compare_i);//set all changed indexes to 0xFFFFFFFF, 1 latency
    __m256i xor_factor = _mm256_xor_si256(compare_i,allONE);//1 latency
    __m256i vdim = _mm256_set1_epi32(dim);
    best_idx_neg = _mm256_add_epi32(best_idx_neg,vdim);//+dim if negativ
    __m256i and_factor = _mm256_or_si256(xor_factor,best_idx_neg);//1 latency
    best_idx = _mm256_and_si256(best_idx,and_factor);//set new best indexes
    best = _mm256_max_ps(best,best_neg);//set new best values


    //bigger floats are still bigger when compare as ints
    __m256i best_i = _mm256_castps_si256(best);
    int value0 = _mm256_extract_epi32(best_i, 0);
    int value1 = _mm256_extract_epi32(best_i, 1);
    int value2 = _mm256_extract_epi32(best_i, 2);
    int value3 = _mm256_extract_epi32(best_i, 3);
    int value4 = _mm256_extract_epi32(best_i, 4);
    int value5 = _mm256_extract_epi32(best_i, 5);
    int value6 = _mm256_extract_epi32(best_i, 6);
    int value7 = _mm256_extract_epi32(best_i, 7);
    int idx0 = _mm256_extract_epi32(best_idx, 0);
    int idx1 = _mm256_extract_epi32(best_idx, 1);
    int idx2 = _mm256_extract_epi32(best_idx, 2);
    int idx3 = _mm256_extract_epi32(best_idx, 3);
    int idx4 = _mm256_extract_epi32(best_idx, 4);
    int idx5 = _mm256_extract_epi32(best_idx, 5);
    int idx6 = _mm256_extract_epi32(best_idx, 6);
    int idx7 = _mm256_extract_epi32(best_idx, 7);

    //for maximum ilp, tree structure of compare
    int res2, res4, res6;
    if(value0>value1){//latency 4
        res = idx0;
    }else{
        res = idx1;
        value0 = value1;
    }
    if(value2>value3){
        res2 = idx2;
    }else{
        res2 = idx3;
        value2 = value3;
    }
    if(value4>value5){
        res4 = idx4;
    }else{
        res4 = idx5;
        value4 = value5;
    }
    if(value6>value7){
        res6 = idx6;
    }else{
        res6 = idx7;
        value6 = value7;
    }
    if(value0>value2){//latency 4
    }else{
        res = res2;
        value0 = value2;
    }
    if(value4>value6){
    }else{
        res4 = res6;
        value4 = value6;
    }
    if(value0>value4){//latency 4
    }else{
        res = res4;
    }
    return res;
}


//minimal runtime: k * locality_sensitive_hash_optimized = k * 12 / 8 * dim
void crosspolytope(float *x, unsigned int *result, int result_size) {
  for(int i = 0; i < result_size; i++){
    result[i]=0;
    int cldim = (int)ceil(log2(num_dimensions))+1;
    for(int ii = 0; ii<k;ii++){
        result[i]<<=cldim;
        result[i]|= locality_sensitive_hash_optimized(&x[ii * num_dimensions], num_dimensions);
    }
  }
}

void random_rotation_precomputed(float *x, int table_idx, int hash_rotation_idx, float *rotated_x) {
    for(int i = 0;i<HMatVecLen;i++){
        float temp = 0;
        for(int ii = 0; ii<HMatVecLen;ii++){
            temp+=x[ii]*RotMat[table_idx * k * HMatVecLen * HMatVecLen
                                       + hash_rotation_idx * HMatVecLen * HMatVecLen
                                       + i*HMatVecLen+ii];
        }
        rotated_x[i] = temp;
    }
}

void random_rotation_precomputed_vectorized(float *x, int table_idx, int hash_rotation_idx, float *rotated_x) {
    for(int i = 0;i<HMatVecLen;i+=8){
        __m256 vtemp = _mm256_setzero_ps();
        float * pos = &RotMat[table_idx * k * HMatVecLen * HMatVecLen
                              + hash_rotation_idx * HMatVecLen * HMatVecLen
                              + i*HMatVecLen];
        for(int ii = 0; ii<HMatVecLen;ii++){
            __m256 vx = _mm256_set1_ps(x[ii]);

            __m256 vRotMat = _mm256_set_ps(pos[ii], pos[HMatVecLen+ii],pos[2*HMatVecLen+ii],pos[3*HMatVecLen+ii],
                                           pos[4*HMatVecLen+ii],pos[5*HMatVecLen+ii],pos[6*HMatVecLen+ii],pos[7*HMatVecLen+ii]);
            vtemp = _mm256_fmadd_ps(vx,vRotMat,vtemp);
        }
        _mm256_storeu_ps(rotated_x+i, vtemp);
    }
}


//overall min runtime over all:     dim / 2 (loading x from RAM)
//                                  + dim * dim / 16 (loading Rotmat from L1
// for dim*dim*k>2^15 (dim>7) even L2/L3,
// L2 means more latency on the loads,
// and L3 means the /16 becomes /8,
// dim * dim floats with beta = 64 B/c)
//                                  + 14 (adding up rows)
//                                  + dim / 2 (storing x)
//                                  = dim * dim / 16 + dim + 14 ( for big dim: dim * dim / 8 + dim + 14)
//overall flops:                    2 * dim * dim
//performance (for big dim):        16 flops/cycle
//performance (for dim = 8):        128/26 = ~5 flops/cycle
//performance (for dim = 16):       512/46 = ~11 flops/cycle
//performance (for dim = 32):       2048/110 = ~19 flops/cycle
//performance (for dim = 64):       8192/334 = ~24.5 flops/cycle
void random_rotation_precomputed_vectorized_unrolled2(float *x, int table_idx, int hash_rotation_idx, float *rotated_x) {
    float * pos = &RotMat[table_idx * k * HMatVecLen * HMatVecLen
                          + hash_rotation_idx * HMatVecLen * HMatVecLen];
    //unroll factor: multiple of number of floats in __m256 and
    // bigger than latency of fmadd (6) and
    // power of 2 to have good performance in adding up rows (using hadd)
    for(int i = 0;i<HMatVecLen;i+=8){
        __m256 vx = _mm256_loadu_ps(x);
        __m256 vRotMat = _mm256_loadu_ps(&pos[i*HMatVecLen]);
        __m256 vRotMat1 = _mm256_loadu_ps(&pos[(i+1)*HMatVecLen]);
        __m256 vRotMat2 = _mm256_loadu_ps(&pos[(i+2)*HMatVecLen]);
        __m256 vRotMat3 = _mm256_loadu_ps(&pos[(i+3)*HMatVecLen]);
        __m256 vRotMat4 = _mm256_loadu_ps(&pos[(i+4)*HMatVecLen]);
        __m256 vRotMat5 = _mm256_loadu_ps(&pos[(i+5)*HMatVecLen]);
        __m256 vRotMat6 = _mm256_loadu_ps(&pos[(i+6)*HMatVecLen]);
        __m256 vRotMat7 = _mm256_loadu_ps(&pos[(i+7)*HMatVecLen]);
        __m256 vtemp = _mm256_mul_ps(vx,vRotMat);
        __m256 vtemp1 = _mm256_mul_ps(vx,vRotMat1);
        __m256 vtemp2 = _mm256_mul_ps(vx,vRotMat2);
        __m256 vtemp3 = _mm256_mul_ps(vx,vRotMat3);
        __m256 vtemp4 = _mm256_mul_ps(vx,vRotMat4);
        __m256 vtemp5 = _mm256_mul_ps(vx,vRotMat5);
        __m256 vtemp6 = _mm256_mul_ps(vx,vRotMat6);
        __m256 vtemp7 = _mm256_mul_ps(vx,vRotMat7);
        //unroll factor: number of floats in __m256
        for(int ii = 8; ii<HMatVecLen;ii+=8){
            vx = _mm256_loadu_ps(&x[ii]);
            vRotMat = _mm256_loadu_ps(&pos[i*HMatVecLen+ii]);
            vtemp = _mm256_fmadd_ps(vx,vRotMat,vtemp);//result for rotated_x[i]
            vRotMat1 = _mm256_loadu_ps(&pos[(i+1)*HMatVecLen+ii]);
            vtemp1 = _mm256_fmadd_ps(vx,vRotMat1,vtemp1);//result for rotated_x[i+1]
            vRotMat2 = _mm256_loadu_ps(&pos[(i+2)*HMatVecLen+ii]);
            vtemp2 = _mm256_fmadd_ps(vx,vRotMat2,vtemp2);
            vRotMat3 = _mm256_loadu_ps(&pos[(i+3)*HMatVecLen+ii]);
            vtemp3 = _mm256_fmadd_ps(vx,vRotMat3,vtemp3);
            vRotMat4 = _mm256_loadu_ps(&pos[(i+4)*HMatVecLen+ii]);
            vtemp4 = _mm256_fmadd_ps(vx,vRotMat4,vtemp4);
            vRotMat5 = _mm256_loadu_ps(&pos[(i+5)*HMatVecLen+ii]);
            vtemp5 = _mm256_fmadd_ps(vx,vRotMat5,vtemp5);
            vRotMat6 = _mm256_loadu_ps(&pos[(i+6)*HMatVecLen+ii]);
            vtemp6 = _mm256_fmadd_ps(vx,vRotMat6,vtemp6);
            vRotMat7 = _mm256_loadu_ps(&pos[(i+7)*HMatVecLen+ii]);
            vtemp7 = _mm256_fmadd_ps(vx,vRotMat7,vtemp7);
        }

        __m256 sum0, sum1, sum2, sum01, sum11, sum21;
        __m128 hi, lo, hi1, lo1, hi2, lo2, vy0, vy4;
        sum0 = _mm256_hadd_ps(vtemp,vtemp1);//r0 r0 r1 r1 r0 r0 r1 r1
        sum1 = _mm256_hadd_ps(vtemp2,vtemp3);
        sum2 = _mm256_hadd_ps(sum0,sum1);//r0 r1 r2 r3 r0 r1 r2 r3
        hi = _mm256_extractf128_ps(sum2,1);
        lo = _mm256_castps256_ps128(sum2);
        vy0 = _mm_add_ps(lo,hi);// r0 r1 r2 r3
        sum01 = _mm256_hadd_ps(vtemp4,vtemp5);
        sum11 = _mm256_hadd_ps(vtemp6,vtemp7);
        sum21 = _mm256_hadd_ps(sum01,sum11);
        hi1 = _mm256_extractf128_ps(sum21,1);
        lo1 = _mm256_castps256_ps128(sum21);
        vy4 = _mm_add_ps(lo1,hi1);// r4 r5 r6 r7

        __m256 vy = _mm256_set_m128(vy4,vy0);

        _mm256_storeu_ps(rotated_x+i, vy);
    }
}


//overall min runtime over all:     dim / 2 (loading x from RAM)
//                                  + dim * dim / 16 - 4 (loading Rotmat from L1, dim * dim floats with beta = 64 B/c, keep the first 8 for initializing temp)
//                                  + 14 (adding up rows)
//                                  + dim / 2 (storing x)
//                                  = (dim * dim / 16 - 4) + dim + 14
//overall flops:                    2 * dim * dim
//performance (for big dim):        32 flops/cycle
//performance (for dim = 8):        128/22 = ~5.5 flops/cycle
//performance (for dim = 16):       512/42 = ~12.5 flops/cycle
//performance (for dim = 32):       2048/106 = ~19.5 flops/cycle
//performance (for dim = 64):       8192/330 = ~25 flops/cycle
void random_rotation_precomputed_vectorized_unrolled2_bulked(float *x, int table_idx, int hash_rotation_idx, float *rotated_x, int bulk_factor) {
    float * pos = &RotMat[table_idx * k * HMatVecLen * HMatVecLen
                          + hash_rotation_idx * HMatVecLen * HMatVecLen];
    //unroll factor: multiple of number of floats in __m256 and
    // power of 2 to have good performance in adding up rows (using hadd)
    for (int i = 0; i < HMatVecLen; i += 8) {
        __m256 vRotMat = _mm256_loadu_ps(&pos[i * HMatVecLen]);
        __m256 vRotMat1 = _mm256_loadu_ps(&pos[(i + 1) * HMatVecLen]);
        __m256 vRotMat2 = _mm256_loadu_ps(&pos[(i + 2) * HMatVecLen]);
        __m256 vRotMat3 = _mm256_loadu_ps(&pos[(i + 3) * HMatVecLen]);
        __m256 vRotMat4 = _mm256_loadu_ps(&pos[(i + 4) * HMatVecLen]);
        __m256 vRotMat5 = _mm256_loadu_ps(&pos[(i + 5) * HMatVecLen]);
        __m256 vRotMat6 = _mm256_loadu_ps(&pos[(i + 6) * HMatVecLen]);
        __m256 vRotMat7 = _mm256_loadu_ps(&pos[(i + 7) * HMatVecLen]);
        //load all vRotMat from L1 is 4 cycle (L1 cache bandwidth is 64b/c)
        for(int b = 0; b<bulk_factor;b++) {
            __m256 vx = _mm256_loadu_ps(&x[b*num_dimensions]);//load dimension*4 bytes per rotation
            __m256 vtemp = _mm256_mul_ps(vx, vRotMat);
            __m256 vtemp1 = _mm256_mul_ps(vx, vRotMat1);
            __m256 vtemp2 = _mm256_mul_ps(vx, vRotMat2);
            __m256 vtemp3 = _mm256_mul_ps(vx, vRotMat3);
            __m256 vtemp4 = _mm256_mul_ps(vx, vRotMat4);
            __m256 vtemp5 = _mm256_mul_ps(vx, vRotMat5);
            __m256 vtemp6 = _mm256_mul_ps(vx, vRotMat6);
            __m256 vtemp7 = _mm256_mul_ps(vx, vRotMat7);
            //throughput mul = 2 => 4 cycle (possible during loading)
            //unroll factor: number of floats in __m256
            for (int ii = 8; ii < HMatVecLen; ii += 8) {
                vx = _mm256_loadu_ps(&x[ii+b*num_dimensions]);//load dimension*4 bytes per rotation
                __m256 vRotMati = _mm256_loadu_ps(&pos[i * HMatVecLen + ii]);
                vtemp = _mm256_fmadd_ps(vx, vRotMati, vtemp);//result for rotated_x[i]
                __m256 vRotMat1i = _mm256_loadu_ps(&pos[(i + 1) * HMatVecLen + ii]);
                vtemp1 = _mm256_fmadd_ps(vx, vRotMat1i, vtemp1);//result for rotated_x[i+1]
                __m256 vRotMat2i = _mm256_loadu_ps(&pos[(i + 2) * HMatVecLen + ii]);
                vtemp2 = _mm256_fmadd_ps(vx, vRotMat2i, vtemp2);
                __m256 vRotMat3i = _mm256_loadu_ps(&pos[(i + 3) * HMatVecLen + ii]);
                vtemp3 = _mm256_fmadd_ps(vx, vRotMat3i, vtemp3);
                __m256 vRotMat4i = _mm256_loadu_ps(&pos[(i + 4) * HMatVecLen + ii]);
                vtemp4 = _mm256_fmadd_ps(vx, vRotMat4i, vtemp4);
                __m256 vRotMat5i = _mm256_loadu_ps(&pos[(i + 5) * HMatVecLen + ii]);
                vtemp5 = _mm256_fmadd_ps(vx, vRotMat5i, vtemp5);
                __m256 vRotMat6i = _mm256_loadu_ps(&pos[(i + 6) * HMatVecLen + ii]);
                vtemp6 = _mm256_fmadd_ps(vx, vRotMat6i, vtemp6);
                __m256 vRotMat7i = _mm256_loadu_ps(&pos[(i + 7) * HMatVecLen + ii]);
                vtemp7 = _mm256_fmadd_ps(vx, vRotMat7i, vtemp7);
            }// fmadd thoughput = 2 => 4 cycle per iteration, but min latency of fmadd => 6
            // so more unrolling? no, since processor does it

            __m256 sum0, sum1, sum2, sum01, sum11, sum21;
            __m128 hi, lo, hi1, lo1, hi2, lo2, vy0, vy4;
            sum0 = _mm256_hadd_ps(vtemp, vtemp1);//r0 r0 r1 r1 r0 r0 r1 r1
            sum1 = _mm256_hadd_ps(vtemp2, vtemp3);
            sum2 = _mm256_hadd_ps(sum0, sum1);//r0 r1 r2 r3 r0 r1 r2 r3
            hi = _mm256_extractf128_ps(sum2, 1);
            lo = _mm256_castps256_ps128(sum2);
            vy0 = _mm_add_ps(lo, hi);// r0 r1 r2 r3
            sum01 = _mm256_hadd_ps(vtemp4, vtemp5);
            sum11 = _mm256_hadd_ps(vtemp6, vtemp7);
            sum21 = _mm256_hadd_ps(sum01, sum11);
            hi1 = _mm256_extractf128_ps(sum21, 1);
            lo1 = _mm256_castps256_ps128(sum21);
            vy4 = _mm_add_ps(lo1, hi1);// r4 r5 r6 r7
            //2 hadd latencies (probably 6) + 1 add latency (4)

            __m256 vy = _mm256_set_m128(vy4, vy0);

            _mm256_storeu_ps(rotated_x + i + b*num_dimensions*k, vy);//store dimension*4 bytes per rotation
        }
    }

}

void random_rotation8_precomputed_vectorized_unrolled2_bulked(float *x, int table_idx, int hash_rotation_idx, float *rotated_x, int bulk_factor) {
    float * pos = &RotMat[table_idx * k * HMatVecLen * HMatVecLen
                          + hash_rotation_idx * HMatVecLen * HMatVecLen];
        __m256 vRotMat = _mm256_loadu_ps(pos);
        __m256 vRotMat1 = _mm256_loadu_ps(&pos[HMatVecLen]);
        __m256 vRotMat2 = _mm256_loadu_ps(&pos[(2) * HMatVecLen]);
        __m256 vRotMat3 = _mm256_loadu_ps(&pos[(3) * HMatVecLen]);
        __m256 vRotMat4 = _mm256_loadu_ps(&pos[(4) * HMatVecLen]);
        __m256 vRotMat5 = _mm256_loadu_ps(&pos[(5) * HMatVecLen]);
        __m256 vRotMat6 = _mm256_loadu_ps(&pos[(6) * HMatVecLen]);
        __m256 vRotMat7 = _mm256_loadu_ps(&pos[(7) * HMatVecLen]);
        for(int b = 0; b<bulk_factor;b+=2) {
            __m256 vx = _mm256_loadu_ps(&x[b*num_dimensions]);
            __m256 vx0 = _mm256_loadu_ps(&x[(b+1)*num_dimensions]);
            __m256 vtemp = _mm256_mul_ps(vx, vRotMat);
            __m256 vtemp1 = _mm256_mul_ps(vx, vRotMat1);
            __m256 vtemp2 = _mm256_mul_ps(vx, vRotMat2);
            __m256 vtemp3 = _mm256_mul_ps(vx, vRotMat3);
            __m256 vtemp4 = _mm256_mul_ps(vx, vRotMat4);
            __m256 vtemp5 = _mm256_mul_ps(vx, vRotMat5);
            __m256 vtemp6 = _mm256_mul_ps(vx, vRotMat6);
            __m256 vtemp7 = _mm256_mul_ps(vx, vRotMat7);
            __m256 vtemp0 = _mm256_mul_ps(vx0, vRotMat);
            __m256 vtemp10 = _mm256_mul_ps(vx0, vRotMat1);
            __m256 vtemp20 = _mm256_mul_ps(vx0, vRotMat2);
            __m256 vtemp30 = _mm256_mul_ps(vx0, vRotMat3);
            __m256 vtemp40 = _mm256_mul_ps(vx0, vRotMat4);
            __m256 vtemp50 = _mm256_mul_ps(vx0, vRotMat5);
            __m256 vtemp60 = _mm256_mul_ps(vx0, vRotMat6);
            __m256 vtemp70 = _mm256_mul_ps(vx0, vRotMat7);

            __m256 sum0, sum1, sum2, sum01, sum11, sum21;
            __m128 hi, lo, hi1, lo1, hi2, lo2, vy0, vy4;
            sum0 = _mm256_hadd_ps(vtemp, vtemp1);//r0 r0 r1 r1 r0 r0 r1 r1
            sum1 = _mm256_hadd_ps(vtemp2, vtemp3);
            sum2 = _mm256_hadd_ps(sum0, sum1);//r0 r1 r2 r3 r0 r1 r2 r3
            hi = _mm256_extractf128_ps(sum2, 1);
            lo = _mm256_castps256_ps128(sum2);
            vy0 = _mm_add_ps(lo, hi);// r0 r1 r2 r3
            sum01 = _mm256_hadd_ps(vtemp4, vtemp5);
            sum11 = _mm256_hadd_ps(vtemp6, vtemp7);
            sum21 = _mm256_hadd_ps(sum01, sum11);
            hi1 = _mm256_extractf128_ps(sum21, 1);
            lo1 = _mm256_castps256_ps128(sum21);
            vy4 = _mm_add_ps(lo1, hi1);// r4 r5 r6 r7

            __m256 vy = _mm256_set_m128(vy4, vy0);

            _mm256_storeu_ps(rotated_x + b*num_dimensions*k, vy);

            __m256 sum00, sum10, sum20, sum010, sum110, sum210;
            __m128 hi0, lo0, hi10, lo10, hi20, lo20, vy00, vy40;
            sum00 = _mm256_hadd_ps(vtemp0, vtemp10);//r0 r0 r1 r1 r0 r0 r1 r1
            sum10 = _mm256_hadd_ps(vtemp20, vtemp30);
            sum20 = _mm256_hadd_ps(sum00, sum10);//r0 r1 r2 r3 r0 r1 r2 r3
            hi0 = _mm256_extractf128_ps(sum20, 1);
            lo0 = _mm256_castps256_ps128(sum20);
            vy00 = _mm_add_ps(lo0, hi0);// r0 r1 r2 r3
            sum010 = _mm256_hadd_ps(vtemp40, vtemp50);
            sum110 = _mm256_hadd_ps(vtemp60, vtemp70);
            sum210 = _mm256_hadd_ps(sum010, sum110);
            hi10 = _mm256_extractf128_ps(sum210, 1);
            lo10 = _mm256_castps256_ps128(sum210);
            vy40 = _mm_add_ps(lo10, hi10);// r4 r5 r6 r7

            __m256 vy2 = _mm256_set_m128(vy40, vy00);

            _mm256_storeu_ps(rotated_x + (b+1)*num_dimensions*k, vy2);
        }
    }



//use random_rotation_precomputed_vectorized_unrolled2 for better performance
void random_rotation_precomputed_vectorized_unrolled(float *x, int table_idx, int hash_rotation_idx, float *rotated_x) {
    for(int i = 0;i<HMatVecLen;i+=8){
        __m256 vtemp = _mm256_setzero_ps();
        __m256 vtemp1 = _mm256_setzero_ps();
        __m256 vtemp2 = _mm256_setzero_ps();
        __m256 vtemp3 = _mm256_setzero_ps();
        __m256 vtemp4 = _mm256_setzero_ps();
        __m256 vtemp5 = _mm256_setzero_ps();
        __m256 vtemp6 = _mm256_setzero_ps();
        __m256 vtemp7 = _mm256_setzero_ps();
        float * pos = &RotMat[table_idx * k * HMatVecLen * HMatVecLen
                              + hash_rotation_idx * HMatVecLen * HMatVecLen
                              + i*HMatVecLen];
        for(int ii = 0; ii<HMatVecLen;ii++){
            //ii
            __m256 vx = _mm256_set1_ps(x[ii]);

            __m256 vRotMat = _mm256_set_ps(pos[ii], pos[HMatVecLen+ii],pos[2*HMatVecLen+ii],pos[3*HMatVecLen+ii],
                                           pos[4*HMatVecLen+ii],pos[5*HMatVecLen+ii],pos[6*HMatVecLen+ii],pos[7*HMatVecLen+ii]);
            vtemp = _mm256_fmadd_ps(vx,vRotMat,vtemp);

            ii++;
            __m256 vx1 = _mm256_set1_ps(x[ii]);

            __m256 vRotMat1 = _mm256_set_ps(pos[ii], pos[HMatVecLen+ii],pos[2*HMatVecLen+ii],pos[3*HMatVecLen+ii],
                                            pos[4*HMatVecLen+ii],pos[5*HMatVecLen+ii],pos[6*HMatVecLen+ii],pos[7*HMatVecLen+ii]);
            vtemp1 = _mm256_fmadd_ps(vx1,vRotMat1,vtemp1);

            ii++;
            __m256 vx2 = _mm256_set1_ps(x[ii]);

            __m256 vRotMat2 = _mm256_set_ps(pos[ii], pos[HMatVecLen+ii],pos[2*HMatVecLen+ii],pos[3*HMatVecLen+ii],
                                            pos[4*HMatVecLen+ii],pos[5*HMatVecLen+ii],pos[6*HMatVecLen+ii],pos[7*HMatVecLen+ii]);
            vtemp2 = _mm256_fmadd_ps(vx2,vRotMat2,vtemp2);

            ii++;
            __m256 vx3 = _mm256_set1_ps(x[ii]);

            __m256 vRotMat3 = _mm256_set_ps(pos[ii], pos[HMatVecLen+ii],pos[2*HMatVecLen+ii],pos[3*HMatVecLen+ii],
                                            pos[4*HMatVecLen+ii],pos[5*HMatVecLen+ii],pos[6*HMatVecLen+ii],pos[7*HMatVecLen+ii]);
            vtemp3 = _mm256_fmadd_ps(vx3,vRotMat3,vtemp3);

            ii++;
            __m256 vx4 = _mm256_set1_ps(x[ii]);

            __m256 vRotMat4 = _mm256_set_ps(pos[ii], pos[HMatVecLen+ii],pos[2*HMatVecLen+ii],pos[3*HMatVecLen+ii],
                                            pos[4*HMatVecLen+ii],pos[5*HMatVecLen+ii],pos[6*HMatVecLen+ii],pos[7*HMatVecLen+ii]);
            vtemp4 = _mm256_fmadd_ps(vx4,vRotMat4,vtemp4);

            ii++;
            __m256 vx5 = _mm256_set1_ps(x[ii]);

            __m256 vRotMat5 = _mm256_set_ps(pos[ii], pos[HMatVecLen+ii],pos[2*HMatVecLen+ii],pos[3*HMatVecLen+ii],
                                            pos[4*HMatVecLen+ii],pos[5*HMatVecLen+ii],pos[6*HMatVecLen+ii],pos[7*HMatVecLen+ii]);
            vtemp5 = _mm256_fmadd_ps(vx5,vRotMat5,vtemp5);

            ii++;
            __m256 vx6 = _mm256_set1_ps(x[ii]);

            __m256 vRotMat6 = _mm256_set_ps(pos[ii], pos[HMatVecLen+ii],pos[2*HMatVecLen+ii],pos[3*HMatVecLen+ii],
                                            pos[4*HMatVecLen+ii],pos[5*HMatVecLen+ii],pos[6*HMatVecLen+ii],pos[7*HMatVecLen+ii]);
            vtemp6 = _mm256_fmadd_ps(vx6,vRotMat6,vtemp6);

            ii++;
            __m256 vx7 = _mm256_set1_ps(x[ii]);

            __m256 vRotMat7 = _mm256_set_ps(pos[ii], pos[HMatVecLen+ii],pos[2*HMatVecLen+ii],pos[3*HMatVecLen+ii],
                                            pos[4*HMatVecLen+ii],pos[5*HMatVecLen+ii],pos[6*HMatVecLen+ii],pos[7*HMatVecLen+ii]);
            vtemp7 = _mm256_fmadd_ps(vx7,vRotMat7,vtemp7);
        }
        vtemp = _mm256_add_ps(vtemp,vtemp1);
        vtemp2 = _mm256_add_ps(vtemp2,vtemp3);
        vtemp4 = _mm256_add_ps(vtemp4,vtemp5);
        vtemp6 = _mm256_add_ps(vtemp6,vtemp7);
        vtemp = _mm256_add_ps(vtemp,vtemp2);
        vtemp4 = _mm256_add_ps(vtemp4,vtemp6);
        vtemp = _mm256_add_ps(vtemp,vtemp4);
        _mm256_storeu_ps(rotated_x+i, vtemp);
    }
}

void random_rotation(float *x, int table_idx, int hash_rotation_idx, int rotation_idx, float *rotated_x) {
    //if(x_size != random_vector.size()||x.size()!=rotated_x.size())
    //    return;//TODO probably should throw error
    //find next smaller power of 2 for hadamard pseudo random rotation

    int log_dim = (int)floor(log2(num_dimensions));
    int h_dim = 1<<log_dim;
    if (h_dim != HMatVecLen) {
      SetHMatVecC(num_dimensions);
    }
    //hadamard transform, in O(n^2), but can be done in O(n log(n)) and falconn does it that way
    for(int i = 0;i<h_dim;i++){
        for(int ii = 0; ii< h_dim; ii++){
          rotated_x[i] += x[ii]*HMatVecC[i&ii];
        }
    }
    float *random_vector = &rotation_vecs[table_idx * k * num_rotations * num_dimensions
                                         + hash_rotation_idx * num_rotations * num_dimensions
                                         + rotation_idx * num_dimensions];
    for(int i = 0; i < num_dimensions; i++){
        rotated_x[i] *= random_vector[i];
    }
}


//runtime per query: k * random_rotation_precomputed_vectorized_unrolled2 =
// k * ((dim * dim / 16) + dim + 14)
void rotations_precomputed(int table_idx, float *data_point, float *result_vec) {
    for(int j = 0;j<k;j++) {
        random_rotation_precomputed_vectorized_unrolled2(data_point, table_idx, j, &result_vec[j*num_dimensions]);
    }
}


//runtime per query: k * random_rotation_precomputed_vectorized_unrolled2_bulked =
// k * ((dim * dim / 16) + dim + 10)
void rotations_precomputed_bulked(int table_idx, float *data_point, float *result_vec, int bulk_factor) {
    /*if(num_dimensions==8){
        for (int j = 0; j < k; j++) {
            random_rotation8_precomputed_vectorized_unrolled2_bulked(data_point, table_idx, j,
                                                                    &result_vec[j * num_dimensions], bulk_factor);
        }
    }else {*/
        for (int j = 0; j < k; j++) {
            random_rotation_precomputed_vectorized_unrolled2_bulked(data_point, table_idx, j,
                                                                    &result_vec[j * num_dimensions], bulk_factor);
        }
    //}
}

void rotations(int table_idx, float *data_point, float *result_vec) {
  float rotated_data[num_dimensions];
  for(int j = 0;j<k;j++) {
    for (int dim = 0; dim < num_dimensions; dim++) {
      result_vec[j*num_dimensions + dim] = data_point[dim];
    }
    for(int r = 0; r < num_rotations; r++){
        for (int dim = 0; dim < num_dimensions; dim++) {
          rotated_data[dim] = result_vec[j*num_dimensions + dim];
          result_vec[j*num_dimensions + dim] = 0;
        }
        random_rotation(rotated_data, table_idx, j, r,
                        &result_vec[j*num_dimensions]);
    }
  }
}


//load 2 * dim * 4 bytes = dim cycles
//runtime 4 + 6 / 8 * (dim-1) + 15
//minimal runtime: load time + 15 = dim + 15 cycles
float negative_inner_product(float * vec1, float * vec2){
    __m256 sv1 = _mm256_loadu_ps(vec1);
    __m256 sv2 = _mm256_loadu_ps(vec2);
    __m256 result = _mm256_mul_ps(sv1,sv2);//latency 4
    for(int i = 8; i < num_dimensions;i+=8){
        __m256 v1 = _mm256_loadu_ps(vec1+i);
        __m256 v2 = _mm256_loadu_ps(vec2+i);
        result = _mm256_fmadd_ps(v1,v2,result);//latency 6
    }
    __m256 sum0, sum2;
    __m128 hi, lo, vy;
    sum0 = _mm256_hadd_ps(result, result);//latency 6 (probably)
    sum2 = _mm256_hadd_ps(sum0, sum0);//latency 6
    hi = _mm256_extractf128_ps(sum2, 1);
    lo = _mm256_castps256_ps128(sum2);
    vy = _mm_add_ps(lo, hi);//latency 3
    return _mm_cvtss_f32(vy);
}

void cleanup(){
    free(RotMat);
    free(HMatVecC);
    free(tables);
    free(rotation_vecs);
}

void print_random_rotation(int table_idx, int hash_idx){
    float * rot = &RotMat[table_idx*k*HMatVecLen*HMatVecLen+hash_idx*HMatVecLen*HMatVecLen];
    printf("Start printing rotation %i, %i\n", table_idx, hash_idx);
    for(int i = 0; i<HMatVecLen;i++){
        for(int ii = 0; ii<HMatVecLen;ii++){
            printf("%f, ", rot[i*HMatVecLen+ii]);
        }
        printf("\n");
    }
}
