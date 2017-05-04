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
