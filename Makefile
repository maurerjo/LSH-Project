CC = gcc
CPP = g++
OBJS = build/lsh.o

all: $(OBJS) main

clean:
	rm build/*

main:
	$(CPP) src/main.cc -std=c++11 -O3 -o build/lsh-project
	
build/lsh.o: src/lsh.h src/lsh.c
	$(CC) -march=native -c src/lsh.h src/lsh.c