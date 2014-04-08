## Makefile                                                                                                                                                             
CC=gcc
CFLAGS= -g#-O3# -pthread
 
HOST:=$(shell hostname | cut -d . -f 1)

ifeq ($(HOST), skele)
INC=-I. -I$(TEMPO2)/include -L$(TEMPO2)/lib -ltempo2 -lblas -llapack -lgsl -lgslcblas -lm -lgomp
INCCULA=-I$(TEMPO2)/include -I. -I/usr/local/cula/include -L/home/pbrem/Downloads/ATLAS/lib/Linux_UNKNOWNSSE2_4 -L/usr/local/cuda-5.5/lib -L/usr/local/cula/lib -lcula_lapack -lcublas -lgsl -L$(TEMPO2)/lib -ltempo2 -lcudart  -lcublas -lcula_lapack -lgsl -lgslcblas -lm -lgomp
endif
 
ifeq ($(HOST), pinatubo)
INC=-I$(TEMPO2)/include -I. -L$(TEMPO2)/lib -ltempo2 -lblas -llapack -lgsl -lgslcblas -lm -lgomp
INCCULA=-I$(TEMPO2)/include -I. -I/home/pbrem/cula/include -L/home/pbrem/cula/lib64 -L$(TEMPO2)/lib -ltempo2 -lcudart  -lcublas -lcula_lapack -lgsl -lgslcblas -lm -lgomp
endif
 
ifeq ($(HOST), krakatoa)
INC=-I$(TEMPO2)/include -I. -L$(TEMPO2)/lib -ltempo2 -lblas -llapack -lgsl -lgslcblas -lm -lgomp
INCCULA=-I$(TEMPO2)/include -I. -I/home/pbrem/cula/include -L/home/pbrem/cula/lib64 -L$(TEMPO2)/lib -ltempo2 -lcudart  -lcublas -lcula_lapack -lgsl -lgslcblas -lm -lgomp
endif

all: main.c
	$(CC) $(CFLAGS) -o pca $^ $(INC)
upper: main.c
	$(CC) $(CFLAGS) -o pca_upper $^ $(INC) -lblas -DUPPER
uppercula: main.c
	$(CC) $(CFLAGS) -o pca_upper $^ $(INCCULA) -lblas -DUPPER -DCULA
f0: main.c
	$(CC) $(CFLAGS) -o pca_F0 $^ $(INC) -lblas -DF0
f0cula: main.c
	$(CC) $(CFLAGS) -o pca_F0 $^ $(INCCULA) -lblas -DF0 -DCULA
fp: main.c
	$(CC) $(CFLAGS) -o pca_Fp $^ $(INC) -lblas
fpcula: main.c
	$(CC) $(CFLAGS) -o pca_Fp $^ $(INCCULA) -lblas -DCULA
cula: main.c
	$(CC) $(CFLAGS) -o pca $^ $(INCCULA) -lblas -DCULA
mpi: main.c
	mpicc $(CFLAGS) -o pca $^ $(INC) -DMPI
culampi: main.c
	mpicc $(CFLAGS) -o pca_cula_mpi $^ $(INCCULA) -DMPI -lblas -DCULA
