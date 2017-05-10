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
void random_rotation(float *x, int x_size, float  *random_vector, float *rotated_x);
void rotations(int dimension, int num_rotation, float *random_rotation_vec, int i,
          float *data_vec, int data_vec_size);
void SetData(float* data_pointer, int points, int dimensions);
void SetTables(int num_tables, int table_size);
void SetRotationVecs(int num_tables, int num_rotations, int k, int num_dimensions);
void init_rng();

#endif /* LSH_H_ */
