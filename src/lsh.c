/*
 * lsh.c
 *
 *  Created on: May 4, 2017
 *      Author: jonathan
 */

#include "lsh.h"

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
