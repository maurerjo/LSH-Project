#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <random>
#include <vector>
#include <algorithm>

using namespace std;
 
int seed = 49628583;
mt19937_64 gen(seed);


/*generate random data
 * @size number of data points genetated
 * @dimension number of dimensions of each data point
 * */
void createData(int size, int dimension, std::vector<float> &data){

	normal_distribution<float> dist_normal(0.0, 1.0);
	for(int i = 0;i < size; i++){
		for(int k = 0; k < dimension; k++){
			data[i*dimension+k]=dist_normal(gen);
		}
	}
}

void createQueries(int num_queries, int dimension, vector<float> &queries){
	createData(num_queries, dimension, queries);
}

void findNearestNeighbours(int size, int dimension, int num_queries, vector<float> &data, vector<float> &queries, vector<int> &nnIDs){
	int nnID;
	float distance;
	for(int i = 0;i<num_queries;i++){
		nnID = 0;
		distance = 0.;
		for(int ii = 0;ii<dimension;ii++){
			distance += (queries[i*dimension+ii]-data[ii])*(queries[i*dimension+ii]-data[ii]);
		}
		for(int k = 1;k<size;k++){
			float current_distance = 0;
			for(int ii = 0; ii<dimension;ii++){
				current_distance += (queries[i*dimension+ii]-data[k*dimension+ii])*(queries[i*dimension+ii]-data[k*dimension+ii]);
			}
			if(current_distance<distance){
				distance = current_distance;
				nnID = k;
			}
		}
		nnIDs[i]=nnID;
		//cout << i <<": " << nnID << ", "<<distance<<endl;
	}
}

//the same as decodeCP of falconn, for comparability
static int locality_sensitive_hash(const vector<float> &data, int dim) {
    int res = 0;
    float best = data[0];
    if (-data[0] > best) {
        best = -data[0];
        res = dim;
    }
    for (int_fast64_t ii = 1; ii < dim; ++ii) {
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

void crosspolytope(vector<float> &x, int k, int dimension, int num_table, int num_rotation, vector<int> &result){
    vector<float> rotated_x(x.size());
    for(int i = 0; i < result.size();i++){
        result[i]=0;
        for(int ii = 0; ii<k-1;ii++){
            result[i]<<=(int)ceil(log2(dimension))+1;
            result[i]|= locality_sensitive_hash(x, dimension);
        }
        result[i]<<=(int)ceil(log2(dimension))+1;
        result[i]|= locality_sensitive_hash(x, dimension);
    }
}

void random_rotation(vector<float> &x, vector<float>  &random_vector, vector<float> &rotated_x){
    if(x.size()!=random_vector.size()||x.size()!=rotated_x.size())
        return;//TODO probably should throw error
    //find next smaller power of 2 for hadamard pseudo random rotation
    int log_dim = (int)floor(log2(x.size()));
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
    for(int i = 0;i<x.size();i++){
        rotated_x[i] = x[i]*random_vector[i];
    }
}

int main(){
    cout << "start\n";
    int size = (1<<10);
    int dimension = 128;
    vector<float> data(size*dimension);
    cout << "create Data Set:\n"<<size<<" data points\n"<<dimension<<" dimensions\n";
    createData(size, dimension, data);
    cout << "finished creating data\n\n";
    int num_queries = 100;
    vector<float> queries(num_queries*dimension);
    cout << "create "<<num_queries<<" queries\n";
    createQueries(num_queries, dimension, queries);
    cout << "finished creating queries\n\n";
    vector<int> nnIDs(num_queries);
    cout << "calculate nearest neighbour via linear scan\n";
    findNearestNeighbours(size, dimension, num_queries, data, queries, nnIDs);
    cout << "found nearest neighbour\n";
    //cross polytope
    cout << "Cross polytope hash" << endl;
    //cross polytope parameters
    int k=3, num_table=10, num_rotation=3;
    //setup tables
    cout << "Create Tables" << endl;
    int table_size = 1<<10;
    vector<vector<float> > tables(num_table);
    vector<vector<vector<float> > > random_rotation_vec(num_table);
    uniform_int_distribution<int> random_bit(0, 1);
    for(int i = 0; i < num_table;i++){
        vector<float> table(table_size);
        tables[i]=table;
        vector< vector<float> >random_rotation(num_rotation);
        for(int r = 0;r<num_rotation;r++){
            vector<float> random_vec(dimension);
            for(int ii = 0; ii < dimension; ii++){
                random_vec[ii]=(float)random_bit(gen);
            }
            random_rotation[r]=random_vec;
        }
        random_rotation_vec[i]=random_rotation;
    }
    cout << "Setup Tables" << endl;
    for(int i = 0; i<num_table;i++){
        for(int ii = 0; ii < size; ii++){
            for(int r = 0; r < num_rotation;r++){
                vector<float>::const_iterator first = data.begin() + r*dimension;
                vector<float>::const_iterator last = data.begin() + (r+1)*dimension;
                vector<float> data_vec(first, last);
            }
        }
    }
    cout << "Finished Table Setup" << endl;
    cout << "Start queries" << endl;
    vector<int> cp_result(num_queries);
    //crosspolytope(k, dimension, num_table, num_rotation, cp_result);
    cout << "Finished queries" << endl;
    int correct_nnIDs=0;
    for(int i = 0; i< num_queries;i++){
        if(cp_result[i]==nnIDs[i]){
            correct_nnIDs++;
        }
    }
    cout << ((float)correct_nnIDs)/((float)num_queries) << "% neighbours found"<<endl;
    return 0;
}
