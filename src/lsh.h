/*
 * lsh.h
 *
 *  Created on: May 4, 2017
 *      Author: jonathan
 */

#ifndef LSH_H_
#define LSH_H_

int locality_sensitive_hash(float *data, int dim);
void crosspolytope(float *x, int k, int dimension, int *result, int result_size);

#endif /* LSH_H_ */
