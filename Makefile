CC = gcc
CPP = g++
OBJS = build/lsh.o

all: $(OBJS) main

main:
	$(CPP) src/main.cc -std=c++11 -o build/lsh-project
	
build/lsh.o: src/lsh.h src/lsh.c
	$(CC) -c src/lsh.h src/lsh.c