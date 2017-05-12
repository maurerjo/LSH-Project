CC = gcc
CPP = g++
OBJS = build/lsh.o

all: $(OBJS) main

clean:
	rm -rf build/*

main:
	$(CPP) src/main.cc src/lsh.c -std=c++11 -flto -march=native -O3 -o build/lsh-project
	
build/lsh.o: src/lsh.h src/lsh.c
	$(CC) -flto -march=native -c src/lsh.h src/lsh.c
