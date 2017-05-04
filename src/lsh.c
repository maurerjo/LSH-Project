/*
 * lsh.c
 *
 *  Created on: May 4, 2017
 *      Author: jonathan
 */

#include "lsh.h"
#include <math.h>

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
  }
}

void random_rotation(float *x, int x_size, float  *random_vector, float *rotated_x) {
    //if(x_size != random_vector.size()||x.size()!=rotated_x.size())
    //    return;//TODO probably should throw error
    //find next smaller power of 2 for hadamard pseudo random rotation
    int log_dim = (int)floor(log2(x_size));
    int h_dim = 1<<log_dim;
    //hadamard scalar
    float scalar = pow(2,-(log_dim/2.0));
    //hadamard transform, in O(n^2), but can be done in O(n log(n)) and falconn does it that way
    for(int i = 0;i<h_dim;i++){
        for(int ii = 0; ii< h_dim; ii++){
            rotated_x[i] += x[ii]*pow(-1,i*ii);
        }
        rotated_x[i]*=scalar;
    }
    for(int i = 0; i<x_size; i++){
        rotated_x[i] *= random_vector[i];
    }
}

void rotations(int dimension, int num_rotation, float *random_rotation_vec, int i,
          float *data_vec, int data_vec_size) {
    for(int r = 0; r < num_rotation; r++){
        float rotated_data[dimension];
        random_rotation(data_vec, data_vec_size, &random_rotation_vec[i*num_rotation*dimension+r],rotated_data);
        data_vec = rotated_data;
    }
}
