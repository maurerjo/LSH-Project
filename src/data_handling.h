/*
 * data_handling.h
 *
 *  Created on: May 19, 2017
 *      Author: jonathan
 */

#ifndef DATA_HANDLING_H_
#define DATA_HANDLING_H_

#include <vector>
#include <string>
#include <fstream>
#include <iostream>

namespace data {

void SaveData(std::string filename, std::vector<float> data, int dimensions) {
  std::cout << "Saving data to file!" << std::endl;
  std::ofstream file(filename);
  int num_points = data.size() / dimensions;
  file << num_points << " " << dimensions << std::endl;
  for (int i = 0; i  < num_points; i++) {
    for (int j = 0; j < dimensions; j++) {
      file << data[i * dimensions + j] << " ";
    }
    file << std::endl;
  }
  std::cout << "Wrote data! Flushing..." << std::endl;
  file.flush();
  file.close();
  std::cout << "File closed!" << std::endl;
}

std::vector<float> LoadData(std::string filename, int &n, int &dimension) {
  std::cout << "Loading data from file!" << std::endl;
  std::ifstream file(filename);
  std::vector<float> data;
  int num_points, dimensions;
  file >> num_points >> dimensions;
  dimension = dimensions;
  n = num_points;
  data.resize(num_points * dimensions);
  for (int i = 0; i < num_points * dimensions; i++) {
    file >> data[i];
  }
  file.close();
  std::cout << "File closed!" << std::endl;
  return data;
}


}




#endif /* DATA_HANDLING_H_ */
