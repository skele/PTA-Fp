CC=gcc
CFLAGS= -g#-pg#-O3# -pthread
#INC=-I. -L/home/pbrem/Downloads/ATLAS/lib/Linux_UNKNOWNSSE2_4 -L/usr/local/cuda-5.5/lib -lgsl -lcblas -latlas -llapack
INC=-I$(TEMPO2)/include -I. -L$(TEMPO2)/lib -ltempo2 -lblas -llapack -lgsl -lgslcblas -lm -lgomp
INCCULA=-I$(TEMPO2)/include -I. -I/home/pbrem/cula/include -L/home/pbrem/cula/lib64 -L$(TEMPO2)/lib -ltempo2 -lcudart  -lcublas -lcula_lapack -lgsl -lgslcblas -lm -lgomp
#INCCULA = -I. -I/usr/local/cula/include -L/home/pbrem/Downloads/ATLAS/lib/Linux_UNKNOWNSSE2_4 -L/usr/local/cuda-5.5/lib -L/usr/local/cula/lib -lcula_lapack -lcublas -lgsl -lcblas -latlas -lgomp
#INC=-I. -I/usr/local/cula/include -L/usr/local/cuda-5.5/lib -L/usr/local/cula/lib -lcula_lapack -lcublas# -lgsl -lcblas -latlas
#INC=-I. -lgsl -lgslcblas
all: main.c
	$(CC) $(CFLAGS) -o pca $^ $(INC)
cula: main.c
	$(CC) $(CFLAGS) -o pca $^ $(INCCULA) -lblas -DCULA
mpi: main.c
	mpicc $(CFLAGS) -o pca $^ $(INC) -DMPI
culampi: main.c
	mpicc $(CFLAGS) -o pca_cula_mpi $^ $(INCCULA) -DMPI -lblas -DCULA