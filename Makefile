CC = gcc
CPP = g++
OBJS = build/lsh.o
#OPTF = -flto -march=native -O2
OPTF = -flto -march=native -Ofast -funsafe-math-optimizations

all: $(OBJS) main

clean:
	rm -rf build/*

main:
	$(CPP) src/main.cc src/lsh.c -std=c++11 $(OPTF) -o build/lsh-project
	
build/lsh.o: src/lsh.h src/lsh.c
	$(CC) -flto $(OPTF) -c src/lsh.h src/lsh.c

build/code_generator.o: src/code_generator.h src/code_generator.c
	$(CC) $(OPTF) -c src/code_generator.h src/code_generator.c

s:
	$(CC) -S src/lsh.c $(OPTF) -o src/lsh.s
