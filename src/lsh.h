/*
 * lsh.h
 *
 *  Created on: May 4, 2017
 *      Author: jonathan
 */

#ifndef LSH_H_
#define LSH_H_

int locality_sensitive_hash(float *data, int dim);
void crosspolytope(float *x, unsigned int *result, int result_size);
void random_rotation(float *x, int table_idx, int hash_rotation_idx, int rotation_idx, float *rotated_x);
void rotations(int table_idx, float *data_point, float *result_vec);
void set_table_entry(int table_idx, unsigned int hash, int entry_idx);
int get_neighbor(int table_idx, unsigned int hash);
void SetData(float* data_pointer, int points, int dimensions);
void SetTables(int num_tables, int table_size);
void SetRotationVecs(int num_tables, int num_rotations, int k, int num_dimensions);
void set_rotation_vec_entry(int table_idx, int hash_rotation_idx, int rotation_idx_, int dim_idx, float value);
void init_rng();

void setup_tables();
void run_queries(float * queries, int num_queries, int * result);

void precomputeRotation();
void rotations_precomputed(int table_idx, float *data_point, float *result_vec);
void random_rotation_precomputed(float *x, int table_idx, int hash_rotation_idx, float *rotated_x);

void print_random_rotation(int table_idx, int hash_idx);

#endif /* LSH_H_ */
