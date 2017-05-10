/*
 * lsh.c
 *
 *  Created on: May 4, 2017
 *      Author: jonathan
 */

#include "immintrin.h"
#include "lsh.h"
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

float * HMatVecC;
int HMatVecLen = 0;

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

int locality_sensitive_hash(float *data, int dim) {
  int res = 0;
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
  }
  return res;
}

void crosspolytope(float *x, int k, int dimension, int *result, int result_size) {
  for(int i = 0; i < result_size; i++){
      result[i]=0;
      int cldim = (int)ceil(log2(dimension))+1;
      for(int ii = 0; ii<k-1;ii++){
          result[i]<<=cldim;
          result[i]|= locality_sensitive_hash(x, dimension);
      }
      result[i]<<=cldim;
      result[i]|= locality_sensitive_hash(x, dimension);
      unsigned int a = 5;
      a = _mm_popcnt_u32(a);
  }
}

void random_rotation(float *x, int x_size, float  *random_vector, float *rotated_x) {
    //if(x_size != random_vector.size()||x.size()!=rotated_x.size())
    //    return;//TODO probably should throw error
    //find next smaller power of 2 for hadamard pseudo random rotation

    int log_dim = (int)floor(log2(x_size));
    int h_dim = 1<<log_dim;
    if (h_dim != HMatVecLen) {
      SetHMatVecC(x_size);
    }
    //hadamard transform, in O(n^2), but can be done in O(n log(n)) and falconn does it that way
    for(int i = 0;i<h_dim;i++){
        for(int ii = 0; ii< h_dim; ii++){
          rotated_x[i] += x[ii]*HMatVecC[i&ii];
        }
    }
    for(int i = 0; i<x_size; i++){
        rotated_x[i] *= random_vector[i];
    }
}

/*void rotations(int dimension, int num_rotation, vector<vector<vector<vector<float> > > > &random_rotation_vec, int i,
          vector<float> &data_vec, vector<vector<float> > &result, int k) {
    for(int j = 0;j<k;j++) {
        result[j].assign(data_vec.begin(),data_vec.begin()+data_vec.size());
        for (int r = 0; r < num_rotation; r++) {
            vector<float> rotated_data(dimension, 0);
            random_rotation(result[j], random_rotation_vec[i][r][j], rotated_data);
            result[j] = move(rotated_data);
        }
    }
}*/

void rotations(int dimension, int num_rotation, float *random_rotation_vec, int i,
          float *data_vec, int data_vec_size) {
  for(int j = 0;j<k;j++) {
    for(int r = 0; r < num_rotation; r++){
        float rotated_data[dimension];
        random_rotation(data_vec, data_vec_size, &random_rotation_vec[i*num_rotation*dimension+r],rotated_data);
        for (int i = 0; i < data_vec_size; i++);
        data_vec = rotated_data;
    }
  }
}
