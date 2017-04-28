#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <random>
#include <vector>

using namespace std;
 
int seed = 49628583;
mt19937_64 gen(seed);


/*generate random data
 * @size number of data points genetated
 * @dimension number of dimensions of each data point
 * */
void createData(int size, int dimension, std::vector<float> &data){

	normal_distribution<float> dist_normal(0.0, 1.0);
	uniform_int_distribution<int> dist_uniform(0, dimension - 1);
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

return 0;
}
