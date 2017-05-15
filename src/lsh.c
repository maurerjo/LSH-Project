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

//for pre computation of rotations
float * HMatC;
int HMatDimLen = 0;

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
    int log_dim = (int)floor(log2(num_dimensions));
    int h_dim = 1<<log_dim;
    HMatC = (float *)malloc(h_dim*h_dim*sizeof(float));
    HMatDimLen = h_dim;
    //hadamard scalar
    float scalar = pow(2,-(log_dim/2.0));
    for(int i = 0; i<h_dim; i++){
        for(int ii = 0; ii<h_dim;ii++){
            HMatC[i*h_dim+ii] = scalar * (1 - ((_mm_popcnt_u32(i&ii) & 0x1) << 1));
        }
    }
    //precompute rotations
    //allocate memory
    RotMat = (float *) malloc(num_tables * k * HMatDimLen *HMatDimLen* sizeof(float));
    float * tempRot;
    tempRot = (float *) malloc(HMatDimLen*HMatDimLen* sizeof(float));
    for(int table_idx = 0; table_idx<num_tables;table_idx++){
        for(int hash_rotation_idx = 0; hash_rotation_idx<k;hash_rotation_idx++){

            //initialize rotation_vec to hadamard
            for(int i = 0; i < h_dim*h_dim;i++){
                RotMat[(table_idx*k+hash_rotation_idx)*h_dim*h_dim+i]=HMatC[i];
            }
            for(int rotation_idx = 0; rotation_idx<num_rotations-1;rotation_idx++){
                for(int i = 0; i<h_dim;i++){
                    for(int ii = 0; ii<h_dim;ii++){
                        RotMat[(table_idx*k+hash_rotation_idx)*h_dim*h_dim+i*h_dim+ii]*=rotation_vecs[table_idx * k * num_rotations * num_dimensions
                                                                                         + hash_rotation_idx * num_rotations * num_dimensions
                                                                                         + rotation_idx * num_dimensions
                                                                                         + ii];//or ii
                        tempRot[i*h_dim+ii] = RotMat[(table_idx*k+hash_rotation_idx)*h_dim*h_dim+i*h_dim+ii];
                    }
                }
                print_random_rotation(table_idx,hash_rotation_idx);
                for(int i = 0; i<h_dim;i++){
                    for(int ii = 0; ii<h_dim;ii++) {
                        float temp = 0;
                        for (int i3 = 0; i3 < h_dim; i3++) {
                            temp += tempRot[i * h_dim + i3] * HMatC[ii * h_dim + i3];//hadamard matrix is it's own transform
                        }
                        RotMat[(table_idx * k + hash_rotation_idx) * h_dim * h_dim + i * h_dim + ii] = temp;
                    }
                }
                print_random_rotation(table_idx,hash_rotation_idx);
            }
            for(int i = 0; i<h_dim;i++){
                for(int ii = 0; ii<h_dim;ii++){
                    RotMat[(table_idx*k+hash_rotation_idx)*h_dim*h_dim+i*h_dim+ii]*=rotation_vecs[table_idx * k * num_rotations * num_dimensions
                                                                                                   + hash_rotation_idx * num_rotations * num_dimensions
                                                                                                   + (num_rotations-1) * num_dimensions
                                                                                                   + ii];//or ii
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

int locality_sensitive_hash(float *data, int dim) {
    int res = 0;
    //for(int i = 0;i<20;i++){
    float best = data[0];
    if (-data[0] > best) {
        best = -data[0];
        res = dim;
    }
    for (int ii = 1; ii < dim; ++ii) {
        if (data[ii] > best) {
            best = data[ii];
            res = ii;
        } else if (-data[ii] > best) {
            best = -data[ii];
            res = ii + dim;
        }
    }//}
    return res;
}


//not the bottle neck
int locality_sensitive_hash_optimized(float *data, int dim) {
    int res = 0;
    //float best = data[0];
    __m256 best = _mm256_load_ps(data);
    __m256 ZERO = _mm256_setzero_ps();
    __m256 best_neg = _mm256_sub_ps(ZERO,best);
    __m256i index = _mm256_set_epi32(0,1,2,3,4,5,6,7);
    __m256i iter = _mm256_set1_epi32(8);
    __m256i allONE = _mm256_set1_epi32(-1);
    __m256i best_idx = index;
    __m256i best_idx_neg = index;
    for (int ii = 1; ii < dim; ii+=8) {
        index = _mm256_add_epi32(index,iter);
        __m256 current = _mm256_load_ps(data+ii);
        __m256 current_neg = _mm256_sub_ps(ZERO,current);
        __m256 compare = _mm256_cmp_ps(best, current, 1);
        __m256i compare_i = _mm256_castps_si256(compare);
        best_idx = _mm256_or_si256(best_idx,compare_i);//set all changed indexes to 0xFFFFFFFF
        __m256i xor_factor = _mm256_xor_si256(compare_i,allONE);
        __m256i and_factor = _mm256_or_si256(xor_factor,index);
        best_idx = _mm256_and_si256(best_idx,and_factor);//set new best indexes TODO
        best = _mm256_max_ps(best,current);//set new best values

        __m256 compare_neg = _mm256_cmp_ps(best_neg, current_neg, 1);
        __m256i compare_neg_i = _mm256_castps_si256(compare_neg);
        best_idx_neg = _mm256_or_si256(best_idx_neg,compare_neg_i);//set all changed indexes to 0xFFFFFFFF
        __m256i xor_factor_neg = _mm256_xor_si256(compare_neg_i,allONE);
        __m256i and_factor_neg = _mm256_or_si256(xor_factor_neg,index);
        best_idx_neg = _mm256_and_si256(best_idx_neg,and_factor_neg);//set new best indexes
        best_neg = _mm256_max_ps(best_neg,current_neg);//set new best negative values
    }
    return res;
}

void crosspolytope(float *x, unsigned int *result, int result_size) {
  for(int i = 0; i < result_size; i++){
    result[i]=0;
    int cldim = (int)ceil(log2(num_dimensions))+1;
    for(int ii = 0; ii<k;ii++){
        result[i]<<=cldim;
        result[i]|= locality_sensitive_hash(&x[ii * num_dimensions], num_dimensions);
    }
  }
}

void random_rotation_precomputed(float *x, int table_idx, int hash_rotation_idx, float *rotated_x) {
    for(int i = 0;i<HMatDimLen;i++){
        for(int ii = 0; ii<HMatDimLen;ii++){
            rotated_x[i]+=x[ii]*RotMat[table_idx * k * HMatDimLen * HMatDimLen
                                       + hash_rotation_idx * HMatDimLen * HMatDimLen
                                       + i*HMatDimLen+ii];
        }
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

void rotations_precomputed(int table_idx, float *data_point, float *result_vec) {
    float rotated_data[num_dimensions];
    for(int j = 0;j<k;j++) {
        for (int dim = 0; dim < num_dimensions; dim++) {
            result_vec[j*num_dimensions + dim] = data_point[dim];
        }
        random_rotation_precomputed(rotated_data, table_idx, j,
                            &result_vec[j*num_dimensions]);
    }
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

void print_random_rotation(int table_idx, int hash_idx){
    float * rot = &RotMat[table_idx*k*HMatDimLen*HMatDimLen+hash_idx*HMatDimLen*HMatDimLen];
    printf("Start printing rotation %i, %i\n", table_idx, hash_idx);
    for(int i = 0; i<HMatDimLen;i++){
        for(int ii = 0; ii<HMatDimLen;ii++){
            printf("%f, ", rot[i*HMatDimLen+ii]);
        }
        printf("\n");
    }
}
