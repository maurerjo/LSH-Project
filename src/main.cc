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


void rotations(int dimension, int num_rotation, vector<vector<vector<vector<float> > > > &random_rotation_vec, int i,
          vector<float> &data_vec, vector<vector<float> > &result, int k);

/*generate random data
 * @size number of data points genetated
 * @dimension number of dimensions of each data point
 * */
void createData(int size, int dimension, vector<float> &data){

	normal_distribution<float> dist_normal(0.0, 1.0);
	for(int i = 0;i < size; i++){
        float scalar = 0.0;
		for(int k = 0; k < dimension; k++){
			data[i*dimension+k]=dist_normal(gen);
            scalar+=data[i*dimension+k]*data[i*dimension+k];
		}//normalize
        for(int k = 0; k < dimension; k++){
            data[i*dimension+k]/=sqrt(scalar);
        }
	}
}

void createQueries(int num_queries, int dimension, vector<float> &queries, int size, vector<float> &data){
	createData(num_queries, dimension, queries);
    uniform_int_distribution<int> random_number(0, size-1);
    normal_distribution<float> dist_normal(0.0, 1.0);
    for(int i = 0; i < num_queries;i++){
        int id=random_number(gen);
        for(int ii = 0; ii< dimension; ii++){
            float random_data_point = data[id*dimension+ii];
            float random_noise = dist_normal(gen);
            queries[i*dimension+ii]=random_data_point*0.95+random_noise*0.05;
        }
    }
}

//negative inner product
void findNearestNeighbours(int size, int dimension, int num_queries, vector<float> &data, vector<float> &queries, vector<int> &nnIDs){
	int nnID;
	float distance;
	for(int i = 0;i<num_queries;i++){
		nnID = 0;
		distance = 0.;
		for(int ii = 0;ii<dimension;ii++){
			distance += (queries[i*dimension+ii]*data[ii]);
		}
		for(int k = 1;k<size;k++){
			float current_distance = 0;
			for(int ii = 0; ii<dimension;ii++){
				current_distance += (queries[i*dimension+ii]*data[k*dimension+ii]);
			}
			if(current_distance>distance){
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
    /*float bla=0;
    for(int i = 0; i<dim;i++){
        bla +=data[i]*data[i];
    }
    cout << bla<<endl;*/
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

void crosspolytope(vector<vector<float> > &x, int k, int dimension, vector<int> &result){
    for(int i = 0; i < result.size();i++){
        result[i]=0;
        int cldim = (int)ceil(log2(dimension))+1;
        //cout << x[0] << "r ";
        for(int ii = 0; ii<k;ii++){
            result[i]<<=cldim;
            result[i]|= locality_sensitive_hash(x[ii], dimension);
            //cout << x[ii][0]/*locality_sensitive_hash(x[ii], dimension)*/<<" ";
        }
        //result[i]<<=cldim;
        //result[i]|= locality_sensitive_hash(x, dimension);
        //cout << locality_sensitive_hash(x, dimension)<<" ";
    }
}

void random_rotation(vector<float> &x, vector<float>  &random_vector, vector<float> &rotated_x){
    if(x.size()!=random_vector.size()||x.size()!=rotated_x.size())
        return;//TODO probably should throw error
    //find next smaller power of 2 for hadamard pseudo random rotation
    /*float distance = 0;
    for(int i = 0;i<x.size();i++){
        distance += x[i]*x[i];
    }*/
    //cout << "distance before: "<< distance;
    int log_dim = (int)floor(log2(x.size()));
    int h_dim = 1<<log_dim;
    //hadamard scalar
    float scalar = pow(2,-(log_dim/2.0));
    //hadamard transform, in O(n^2), but can be done in O(n log(n)) and falconn does it that way
    for(int i = 0;i<h_dim;i++){
        for(int ii = 0; ii< h_dim; ii++){
            rotated_x.at(i) += x.at(ii)*pow(-1,__builtin_popcount(i&ii));
        }
        rotated_x[i]*=scalar;
    }
    for(int i = 0;i<x.size();i++){
        rotated_x.at(i) *= random_vector.at(i);
    }

    /*float distance = 0;
    for(int i = 0;i<x.size();i++){
        distance += rotated_x[i]*rotated_x[i];
    }
    for(int i = 0;i<x.size();i++){
        rotated_x[i]/=sqrt(distance);
    }
    cout << "distance after: "<< distance;*/
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
    createQueries(num_queries, dimension, queries, size, data);
    cout << "finished creating queries\n\n";
    vector<int> nnIDs(num_queries);
    cout << "calculate nearest neighbour via linear scan\n";
    findNearestNeighbours(size, dimension, num_queries, data, queries, nnIDs);
    cout << "found nearest neighbour\n";
    //cross polytope
    cout << "Cross polytope hash" << endl;
    //cross polytope parameters
    int k=3, num_table=17, num_rotation=3;
    //setup tables
    cout << "Create Tables" << endl;
    int table_size = (1<<15)-1;
    vector<vector<int> > tables(num_table);
    vector<vector<vector<vector<float> > > > random_rotation_vec(num_table);
    uniform_int_distribution<int> random_bit(0, 1);
    for(int i = 0; i < num_table;i++){
        vector<int> table(table_size);
        tables[i]=move(table);
        vector<vector< vector<float> > >random_rotation(num_rotation);
        for(int r = 0;r<num_rotation;r++){
            vector<vector<float> >random_vec(k);
            for(int ii = 0;ii<k;ii++){
                vector<float>k_vec(dimension);
                for(int i3 = 0; i3 < dimension; i3++){
                    k_vec[i3]=(float)(2*random_bit(gen)-1);
                }
                random_vec[ii] = k_vec;
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

            /*float distance = 0;
            for(int k = 0; k < dimension; k++){
                distance += data_vec[k]*data_vec[k];
            }
            cout << "distance setup_table: "<<distance;*/
            vector<vector<float> > rotations_vec = vector<vector<float> >(k);
            rotations(dimension, num_rotation, random_rotation_vec, i, data_vec, rotations_vec,k);
            vector<int> result(1);
            crosspolytope(rotations_vec,k,dimension,result);
            //cout << result[0]<<" ";
            tables[i][result[0]%table_size] = ii;
        }
    }
    cout << "Finished Table Setup" << endl;
    cout << "Start queries" << endl;
    vector<int> cp_result(num_queries);
    int close = 0;
    for(int ii = 0; ii < num_queries; ii++){
        vector<int> result_vote(num_table);
        vector<int> num_votes(num_table);
        for(int i = 0; i<num_table;i++){
            vector<float>::const_iterator first = queries.begin() + ii*dimension;
            vector<float>::const_iterator last = queries.begin() + (ii+1)*dimension;
            vector<float> query_vec(first, last);
            vector<vector<float> > rotated_query = vector<vector<float> >(k);
            rotations(dimension, num_rotation, random_rotation_vec, i, query_vec, rotated_query,k);
            vector<int> result(1);
            crosspolytope(rotated_query,k,dimension,result);
            //cout <<" "<< result[0]<<" ";
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
                    if(tables[i][result[0]%table_size]==nnIDs[ii]){
                        close++;
                    }
                }
                //cout << i << ", " << ii << ", " << tables[i][result[0] % table_size]<< ", " << nnIDs[ii]<<endl;
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
    int table_used=0;
    for(int i = 0; i < table_size;i++){
        if(tables[0][i]!=0){
            table_used++;
        }
    }
    cout << 100*((float)correct_nnIDs)/((float)num_queries) << "% neighbours found"<<endl;
    cout << 100*((float)close)/((float)num_queries) << "% close found"<<endl;
    cout << table_used << " table entries used"<<endl;
    return 0;
}

void rotations(int dimension, int num_rotation, vector<vector<vector<vector<float> > > > &random_rotation_vec, int i,
          vector<float> &data_vec, vector<vector<float> > &result, int k) {
    for(int j = 0;j<k;j++) {
        result[j].assign(data_vec.begin(),data_vec.begin()+data_vec.size());
        for (int r = 0; r < num_rotation; r++) {
            vector<float> rotated_data(dimension, 0);
            random_rotation(result[j], random_rotation_vec[i][r][j], rotated_data);
            result[j] = move(rotated_data);
        }
    }
}
