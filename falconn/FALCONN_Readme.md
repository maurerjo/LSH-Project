In order to modify FALCONN for tests follow the following steps.

1. copy falcon into the LSH-Project root (FALCONN-master on the same level as src)
2. replace the random_benchmark.cc in the FALCONN library with the modified one from this folder.
3. modify the settings of the LSH structure (k and num_tables) in order to match your preferences.
4. compile the random benchmark with "make random_benchmark" in the FALCONN-master folder
5. run the LSH-Project to generate data in the data folder
6. run the random_benchmark