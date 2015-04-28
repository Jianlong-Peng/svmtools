#USAGE: make
CFLAGS = -Wall -O2

all: grid_search_cgp.exe predict.exe

predict.exe: predict.c svm.o svmtools.o
	g++ $(CFLAGS) predict.c svm.o svmtools.o -o predict.exe -lm

grid_search_cgp.exe: grid_search_cgp.o svm.o svmtools.o
	g++ $(CFLAGS) grid_search_cgp.c svm.o svmtools.o -o grid_search_cgp.exe -lm

svm.o: svm.cpp
	g++ $(CFLAGS) -c svm.cpp

svmtools.o: svmtools.c
	g++ $(CFLAGS) -c svmtools.c -o svmtools.o

svmtools_p.o: svmtools.c
	g++ $(CFLAGS) -DUSE_THREAD -c svmtools.c -o svmtools_p.o
