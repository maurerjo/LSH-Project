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


void rotations(int dimension, int num_rotation, vector<vector<vector<float> > > &random_rotation_vec, int i,
          vector<float> &data_vec);

/*generate random data
 * @size number of data points genetated
 * @dimension number of dimensions of each data point
 * */
void createData(int size, int dimension, std::vector<float> &data){

	normal_distribution<float> dist_normal(0.0, 1.0);
	for(int i = 0;i < size; i++){
    float scalar = 0;
		for(int k = 0; k < dimension; k++){
			data[i*dimension+k]=dist_normal(gen);
        scalar+=data[i*dimension+k]*data[i*dimension+k];
		}//normalize
    for(int k = 0; k < dimension; k++){
      data[i*dimension+k]/=scalar;
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
static int locality_sensitive_hash(vector<float> &data, int dim) {
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

void crosspolytope(vector<float> &x, int k, int dimension, vector<int> &result){
    for(int i = 0; i < result.size();i++){
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
        rotated_x[i] *= random_vector[i];
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
    int table_size = 10000;
    vector<vector<int> > tables(num_table);
    vector<vector<vector<float> > > random_rotation_vec(num_table);
    uniform_int_distribution<int> random_bit(0, 1);
    for(int i = 0; i < num_table;i++){
        vector<int> table(table_size);
        tables[i]=move(table);
        vector< vector<float> >random_rotation(num_rotation);
        for(int r = 0;r<num_rotation;r++){
            vector<float> random_vec(dimension);
            for(int ii = 0; ii < dimension; ii++){
                random_vec[ii]=(float)(2*random_bit(gen)-1);
            }
            random_rotation[r]=move(random_vec);
        }
        random_rotation_vec[i]=move(random_rotation);
    }
    cout << "Setup Tables" << endl;
    for(int i = 0; i<num_table;i++){
        for(int ii = 0; ii < size; ii++){
            vector<float>::const_iterator first = data.begin() + ii*dimension;
            vector<float>::const_iterator last = data.begin() + (ii+1)*dimension;
            vector<float> data_vec(first, last);
            rotations(dimension, num_rotation, random_rotation_vec, i, data_vec);
            vector<int> result(1);
            crosspolytope(data_vec,k,dimension,result);
            tables[i][result[0]%table_size] = ii;
        }
    }
    cout << "Finished Table Setup" << endl;
    cout << "Start queries" << endl;
    vector<int> cp_result(num_queries);
    for(int ii = 0; ii < num_queries; ii++){
        vector<int> result_vote(num_table);
        vector<int> num_votes(num_table);
        for(int i = 0; i<num_table;i++){
            vector<float>::const_iterator first = queries.begin() + ii*dimension;
            vector<float>::const_iterator last = queries.begin() + (ii+1)*dimension;
            vector<float> query_vec(first, last);
            rotations(dimension, num_rotation, random_rotation_vec, i, query_vec);
            vector<int> result(1);
            cout << result[0];
            crosspolytope(query_vec,k,dimension,result);
            cout <<" "<< result[0]<<" ";
            if(tables[i][result[0]%table_size]!=0) {
                bool found = false;
                for(int j = 0; j < i; j++){
                    if(tables[i][result[0]%table_size]==result_vote[j]){
                        num_votes[j]++;
                        found = true;
                    }
                }
                if(!found){
                    result_vote[i]=tables[i][result[0]%table_size];
                }
                cout << ii << ", " << tables[i][result[0] % table_size]<<endl;
            }
        }
        int max = 0;
        for(int i = 0; i<num_table; i++){
            if(num_votes[i]>max){
                cp_result[ii]=result_vote[i];
                max = num_votes[i];
            }
        }
    }
    cout << "Finished queries" << endl;
    int correct_nnIDs=0;
    for(int i = 0; i< num_queries;i++){
        if(cp_result[i]==nnIDs[i]){
            correct_nnIDs++;
        }
    }
    cout << 100*((float)correct_nnIDs)/((float)num_queries) << "% neighbours found"<<endl;
    return 0;
}

void rotations(int dimension, int num_rotation, vector<vector<vector<float> > > &random_rotation_vec, int i,
          vector<float> &data_vec) {
    for(int r = 0; r < num_rotation; r++){
        vector<float> rotated_data(dimension);
        random_rotation(data_vec, random_rotation_vec[i][r],rotated_data);
        data_vec = move(rotated_data);
    }
}
