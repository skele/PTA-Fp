#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
//#include <pthread.h>
#include <gsl/gsl_randist.h>
#include <stdlib.h>
#include <time.h>
#include "tempo2.h"

#ifdef CULA
#include <cula.h>
#endif

#ifdef MPI
#include <mpi.h>
#endif

#define verbose 0
#define NFFT 40
#define N_SAMPLE_MAX 27000
#define MAX_PSR 45
#define MAX_BE 30
#define NF0 100

#include <mydefs.h>

//#define UPPER
#define EFAC
#define DM

#define K_DM 2.41E-4 //in Mhz^-2 pc s^-1

#define pc 3.0856775807e16
#define PI 3.14159
#define c 299792458.0
#define pc_sec 102927125.028
#define Msun 4.92549095e-6
#define year 31556736.0
//#define SYMCHECK
#define NSWARM 2
#define NITER 100
#define NPERNODE 4
//#define MCMC
//#define GRID

//#define PCA

//#define SINGLE
//#define CUNITY

#define REDNOISE
//#define POWERLAW
//#define SWITCHEROO

#define ETA 0.05

void compute_Nwiggle(struct mypulsar * pulsar)
{
  int N = pulsar->N;
  int N_m = pulsar->N_m;
  struct my_matrix * GNG, *GNG_temp, *Cholesky, *GNG_temp2;
  GNG = my_matrix_alloc(N_m,N_m);
  GNG_temp = my_matrix_alloc(N_m,N);

  double tstart = omp_get_wtime();


  int j,k;

#ifdef EFAC
  //draw new data for pulsar from chain
  double efac;
  double equad;
  int ibe;
#endif
  //load up new red noise parameters
#ifdef DM
#ifdef REDNOISE
  pulsar->rA = pow(10.0,pulsar->sample->data[pulsar->index*pulsar->sample->m + 0]);
  pulsar->rgamma = pulsar->sample->data[pulsar->index*pulsar->sample->m + 1];
#endif
  pulsar->dmA = pow(10.0,pulsar->sample->data[pulsar->index*pulsar->sample->m + 2]);
  pulsar->dmgamma = pulsar->sample->data[pulsar->index*pulsar->sample->m + 3];
  if (verbose == 2)
    printf("rednoise %g\t%g\t DM    %g\t%g\n",pulsar->rA,pulsar->rgamma,pulsar->dmA,pulsar->dmgamma);
#endif
  for (j = 0; j < pulsar->N; j++)
    for (k = 0; k < pulsar->N; k++)
      {
	if (j == k)
	  {
#ifdef EFAC
	    ibe = pulsar->backends[j];
	    efac = pulsar->sample->data[pulsar->index*pulsar->sample->m + 4 + ibe];
	    equad = pow(10.0,pulsar->sample->data[pulsar->index*pulsar->sample->m + 4 + pulsar->n_be + ibe]);
	    //printf("efac %d %g\t%g\n",ibe,efac,equad);
	    pulsar->CWN->data[k*pulsar->N+j] = pulsar->sigma[j]*pulsar->sigma[j]*efac*efac + equad*equad;
#else	 
	    pulsar->CWN->data[k*pulsar->N+j] = pulsar->sigma[j]*pulsar->sigma[j];
#endif
	  }
	else
	  pulsar->CWN->data[k*pulsar->N+j] = 0.0;
	//	    gsl_matrix_set(pulsar->CWN,j,j,pulsar->sigma[j]*pulsar->sigma[j]);
      }
#ifdef EFAC
  //  printf("End CWN\n");
#endif

  double tend = omp_get_wtime();
  if (verbose == 3)
    printf("built up CWN\t\t\t\t%g\n",tend-tstart);

  tstart = omp_get_wtime();
  my_dgemm(CblasTrans,CblasNoTrans,1.0,pulsar->G,pulsar->CWN,0.0,GNG_temp);

  my_dgemm(CblasNoTrans,CblasNoTrans,1.0,GNG_temp,pulsar->G,0.0,GNG);
  tend = omp_get_wtime();
  if (verbose == 3)
    printf("built up GNG\t\t\t\t%g\n",tend-tstart);
  //Get inverse
  tstart = omp_get_wtime();
  Cholesky = my_matrix_alloc(N_m,N_m);
  //  my_matrix_print(GNG);
  if (verbose == 2)
    printf("Trying to invert GNG\n");
  double det;
  if (get_inverse_lu(GNG,Cholesky,N_m,&det) == 0 && (verbose == 2))
    printf("My inversion of GNG worked\n");
  my_matrix_memcpy(pulsar->GNGinv,GNG);

  tend = omp_get_wtime();
  if (verbose == 3)
    printf("inverted GNG\t\t\t\t%g\n",tend-tstart);
//  if (verbose)
//    test_inversion(GNG,pulsar->GNG);
  //GNG contains GNG^-1
  //calc G GNG^-1 into 
  
//  tstart = omp_get_wtime();
//
//  GNG_temp2 = my_matrix_alloc(N,N_m);
//  //  my_dgemm(CblasNoTrans,CblasNoTrans,1.0,pulsar->G,GNG,0.0,GNG_temp2);
//  my_dgemm(CblasNoTrans,CblasNoTrans,1.0,pulsar->G,GNG,0.0,GNG_temp2);
//  //calc Nwiggle
//  my_dgemm(CblasNoTrans,CblasTrans,1.0,GNG_temp2,pulsar->G,0.0,pulsar->Nwiggle);
//
//  tend = omp_get_wtime();
//  if (verbose == 3)
//    printf("obtained Nwiggle\t\t\t\t%g\n",tend-tstart);
  //for testing inversion
  //  gsl_matrix * chol2;
//  for (i = 0; i < N; i++)
//    {
//      for (j = i; j < N; j++)
//	{
//	  el1 = gsl_matrix_get(pulsar->Nwiggle,i,j);
//	  el2= gsl_matrix_get(pulsar->Nwiggle,j,i);
//	  printf("%d\t%d\t%g\t%g\t%g\n",i,j,(el1-el2)/(el1+el2),el1,el2);
//	}
//    }
//  gsl_matrix * testing;
//  testing = gsl_matrix_alloc(N_m,N_m);

  //test if GTG = 1
//  my_dgemm(CblasTrans,CblasNoTrans,1.0,pulsar->G,pulsar->G,0.0,testing);
//  int i;
//  for (i = 0; i < N_m; i++)
//    printf("%d\t%g\n",i,gsl_matrix_get(testing,i,i));
//
//  gsl_matrix_free(testing);
//  char header[20] = "TEMPMATRIX\0";
//  //  header = 'FNF';
//  my_matrix_print(header,GNG_temp);
//      
//  char header2[20] = "GNGMATRIX\0";
//  //  header = 'FNF';
//  my_matrix_print(header2,GNG);


  my_matrix_free(Cholesky);
  my_matrix_free(GNG);
  my_matrix_free(GNG_temp);

}

void compute_H(struct mypulsar * psr, struct parameters * par)
{
  int j;      //build up H matrix
  for (j = 0; j < psr->N; j++)
    {
      psr->H->data[0*psr->H->m + j] = sin(par->omega*psr->toa->data[j]);
      psr->H->data[1*psr->H->m + j] = cos(par->omega*psr->toa->data[j]);
    }
  my_dgemm(CblasTrans,CblasNoTrans,1.0,psr->G,psr->H,0.0,psr->GH);

}

void initialize_pulsars(struct mypulsar * pulsars, char ** filenames, int Nplsr, int *Ndim, int *Ntot, struct parameters * par)
{
  int i,j,k;
  FILE *infile;
  double temp;
  //initialize a struct for each pulsar
  *Ndim = 0;
  *Ntot = 0;
  for (i = 0; i < Nplsr; i++)
    {
      infile = fopen(filenames[i],"r");
      if (fscanf(infile,"%le\n%le\n",&(pulsars[i].raj),&(pulsars[i].dec)) != 0);
      if (fscanf(infile,"%d\n",&(pulsars[i].N)) != 0);
      pulsars[i].toa = my_vector_alloc(pulsars[i].N);
      pulsars[i].sigma = (double *) malloc(pulsars[i].N * sizeof(double));
      for (j = 0; j < pulsars[i].N; j++)
	if (fscanf(infile,"%le\n",&(pulsars[i].toa->data[j])) == 0)
	  fprintf(stderr,"Read error toa\n");
      for (j = 0; j < pulsars[i].N; j++)
	if (fscanf(infile,"%le\n",&(pulsars[i].sigma[j])) == 0)
	  fprintf(stderr,"Read error sigma\n");
      if (fscanf(infile,"%d %d %d %d\n",&(pulsars[i].N),&(pulsars[i].N_m),&(pulsars[i].n_be),&(pulsars[i].n_sample)) != 0);
      pulsars[i].res = my_vector_alloc(pulsars[i].N);
      pulsars[i].obsfreqs = my_vector_alloc(pulsars[i].N);
      pulsars[i].backends = (int *) malloc(pulsars[i].N*sizeof(int));
      for (j = 0; j < pulsars[i].N; j++)
	if (fscanf(infile,"%le\n",&(pulsars[i].res->data[j])) == 0)
	  fprintf(stderr,"Read error res\n");
      for (j = 0; j < pulsars[i].N; j++)
	if (fscanf(infile,"%le\n",&(pulsars[i].obsfreqs->data[j])) == 0)
	  fprintf(stderr,"Read error res\n");
      for (j = 0; j < pulsars[i].N; j++)
	if (fscanf(infile,"%d\n",&(pulsars[i].backends[j])) == 0)
	  fprintf(stderr,"Read error res\n");
      //      pulsars[i].G = gsl_matrix_calloc(pulsars[i].N,pulsars[i].N_m);
      //my_matrix_init(pulsars[i].G,pulsars[i].N,pulsars[i].N_m);
      pulsars[i].G = my_matrix_alloc(pulsars[i].N,pulsars[i].N_m);
      for (j = 0; j < pulsars[i].N; j++)
	for (k = 0; k < pulsars[i].N_m; k++)
	  {
	    if (fscanf(infile,"%le\n",&temp) != 0)
	      pulsars[i].G->data[k*pulsars[i].G->m+j] = temp;
      //	      gsl_matrix_set(pulsars[i].G,j,k,temp);
	    else
	      {
		printf("ERROR IN READING G MATRIX\n");
	      }
	  }
      *Ndim += pulsars[i].N_m;
      *Ntot += pulsars[i].N;
      pulsars[i].CWN = my_matrix_alloc(pulsars[i].N,pulsars[i].N);
      //pulsars[i].CWN = gsl_matrix_calloc(pulsars[i].N,pulsars[i].N); //initialize with 0s

#ifdef DM
      pulsars[i].phi_inv = my_matrix_alloc(2*NFFT,2*NFFT);
      pulsars[i].F = my_matrix_alloc(pulsars[i].N,2*NFFT);
      pulsars[i].GF = my_matrix_alloc(pulsars[i].N_m,2*NFFT);
#else
      pulsars[i].phi_inv = my_matrix_alloc(NFFT,NFFT);
      pulsars[i].F = my_matrix_alloc(pulsars[i].N,NFFT);
      pulsars[i].GF = my_matrix_alloc(pulsars[i].N_m,NFFT);
#endif

      pulsars[i].GNGinv =  my_matrix_alloc(pulsars[i].N_m,pulsars[i].N_m);
      pulsars[i].H = my_matrix_alloc(pulsars[i].N,2);

      pulsars[i].C = my_matrix_alloc(pulsars[i].N_m,pulsars[i].N_m);
      pulsars[i].Cinv = my_matrix_alloc(pulsars[i].N_m,pulsars[i].N_m);
      pulsars[i].L = my_matrix_alloc(pulsars[i].N_m,pulsars[i].N_m);
      pulsars[i].GH = my_matrix_alloc(pulsars[i].N_m,2);

      pulsars[i].Gres = my_vector_alloc(pulsars[i].N_m);
      my_dgemv(CblasTrans,1.0,pulsars[i].G,pulsars[i].res,0.0,pulsars[i].Gres);
      //      my_vector_print(pulsars[i].Gres);
      compute_H(&(pulsars[i]),par);

      fclose(infile);
      //read in sampled data
      int xdim = 4+2*pulsars[i].n_be+1;
      pulsars[i].sample = my_matrix_alloc(xdim,pulsars[i].n_sample);
      char infilename[100];
      sprintf(infilename,"%s/chains_Noise/Noise_PlusMono_post_equal_weights.dat",pulsars[i].name);
      infile = fopen(infilename,"r");
      for (j = 0; j < pulsars[i].n_sample; j++)
	for (k = 0; k < xdim; k++)
	  if (fscanf(infile,"%le",&(pulsars[i].sample->data[j*xdim + k])) == 0)
	    fprintf(stderr,"Read error sampling\n");
      if (verbose == 2)
	{
	  printf("first sample line:\n");
	  for (k = 0; k < xdim; k++)
	    printf("%g\t",pulsars[i].sample->data[k]);
	  printf("\n");
	}

      double tmin= 1.0E10;
      double tmax = 0.0;
      //  double temp;
      int b;
      for (b = 0; b < pulsars[i].N; b++)
	{
	  if ((pulsars[i].toa->data[b]) > tmax)
	    tmax = pulsars[i].toa->data[b];
	  if ((pulsars[i].toa->data[b]) < tmin)
	    tmin = pulsars[i].toa->data[b];
	}
      temp = (tmax - tmin);
      //temp = *tspan*1.306;                                                                                                                                                 
      //  temp = *tspan;
      //temp = *tspan*1.18;                                                                                                                                                  
      double ffund = 1.0/(temp);
      //set up frequency grid                                                                                                                                                
      for (b = 0; b < (NFFT/2); b++)
	{
	  pulsars[i].freqs[b] = ffund*(b+1);
	  //      printf("%d\t%e\n",i,par->freqs[i]);
	}
      pulsars[i].tspan = temp;
      
      //      compute_Nwiggle(&(pulsars[i]));

    }


  if (verbose)
    printf("Ndim is %d\n",*Ndim);
}

void get_G_matrix(struct my_matrix * G, pulsar psr, struct mypulsar mypsr)
{
  int j,k;
  double epoch = *(psr.param[param_pepoch].val);
  int ndim = psr.nParam;
  int ma = ndim;
  struct my_matrix * design;
  design = my_matrix_alloc(ma,mypsr.N);
  struct my_matrix * designT;
  designT = my_matrix_alloc(mypsr.N,ma);
  for (j = 0; j < mypsr.N; j++)
    {
      FITfuncs(psr.obsn[j].bat - epoch,&(design->data[j*design->m]),ma,&psr,j,0);
    }
  struct my_vector * singulars;
  struct my_matrix *U;
  U = my_matrix_alloc(mypsr.N,mypsr.N);
  singulars = my_vector_alloc(mypsr.N);
  //transpose designmatrix
  for (j = 0; j < mypsr.N; j++)
    for (k = 0; k < ma; k++)
      designT->data[k*designT->m + j] = design->data[j*design->m+k];
  my_svd(designT,U,singulars);
  //copy last N_m columns
  memcpy(G->data,&(U->data[ma*U->m]),mypsr.N_m*mypsr.N*sizeof(double));

    my_matrix_free(design);
    my_matrix_free(U);
    my_matrix_free(designT);

}

void initialize_pulsars_fromtempo(pulsar * tempo_psrs, struct mypulsar * pulsars, int Nplsr, int *Ndim, int *Ntot, struct parameters * par, int only_res)
{
  FILE *infile;
  int i,j,k,l;
  if (verbose)
    printf("initializing from tempo\n");
  double temp;
  //initialize a struct for each pulsar
  *Ndim = 0;
  *Ntot = 0;
  for (i = 0; i < Nplsr; i++)
    {
      pulsar psr = tempo_psrs[i];
      char backends[MAX_BE][MAX_FLAG_LEN];
      int bid = 0;
      if (only_res == 0) //do all init
	{
	  if (verbose == 3)
	    printf("Initializing for the first time pulsar %d\n",i);
      //dereference tempo pulsar to save writing
	  pulsars[i].n_be=0;
	  pulsars[i].raj = atof(psr.rajStrPost);
	  pulsars[i].dec = atof(psr.decjStrPost);
	  pulsars[i].N = psr.nobs;

	  pulsars[i].toa = my_vector_alloc(pulsars[i].N);
	  pulsars[i].sigma = (double *) malloc(pulsars[i].N * sizeof(double));
	  pulsars[i].oldbat = (double *) malloc(pulsars[i].N * sizeof(double));
	}
      
	  for (j = 0; j < pulsars[i].N; j++)
	    {
	      pulsars[i].toa->data[j] = psr.obsn[j].bat*86400.0;
	      if (only_res == 0)
		pulsars[i].oldbat[j] = psr.obsn[j].sat*86400.0;
	    }
	  if (only_res == 0)
	    {
	      for (j = 0; j < pulsars[i].N; j++)
		pulsars[i].sigma[j] = psr.obsn[j].toaErr * 1e-6;
	  //      pulsars[i].N_m = 
	  //if (fscanf(infile,"%d %d %d %d\n",&(pulsars[i].N),&(pulsars[i].N_m),&(pulsars[i].n_be),&(pulsars[i].n_sample)) != 0);
	      pulsars[i].res = my_vector_alloc(pulsars[i].N);
	      pulsars[i].obsfreqs = my_vector_alloc(pulsars[i].N);
	      pulsars[i].backends = (int *) malloc(pulsars[i].N*sizeof(int));
	    }
      //Do this part always
      for (j = 0; j < pulsars[i].N; j++)
	{
	  pulsars[i].res->data[j] = psr.obsn[j].residual;
	  if (only_res == 0)
	    {
	      pulsars[i].obsfreqs->data[j] = psr.obsn[j].freq;
	      l = 0;
	      //	  if (verbose)
	      //	    printf("Found %d flags\n",psr.obsn[j].nFlags);
	      while ((strcmp(psr.obsn[j].flagID[l],"-sys") != 0) && (l < psr.obsn[j].nFlags))
		{
		  l++;
		}
	      bid = pulsars[i].n_be; //if i dont find the backend already, add a new string
	      for (k = 0; k < pulsars[i].n_be; k++)
		{
		  //check if already sys flag exists in backends
		  if (strcmp(psr.obsn[j].flagVal[l],backends[k]) == 0)
		    bid = k;
		}
	      if (bid == pulsars[i].n_be)
		{
		  if (verbose)
		    printf("Found new backend\t%s\n",psr.obsn[j].flagVal[l]);
		  strcpy(backends[pulsars[i].n_be],psr.obsn[j].flagVal[l]);
		  pulsars[i].n_be++;
		}
	      pulsars[i].backends[j] = bid;
	      //	  pulsars[i].backends[j] = psr.obsn[j].residual;
	    }
	}
      //compute design matrix here and sort backends
      if (only_res == 0)
	{
	  int *sorted;
	  sorted = (int *) malloc(pulsars[i].n_be * sizeof(int));
	  argsort(backends,sorted,pulsars[i].n_be);
	  if (verbose)
	    for (j = 0; j < pulsars[i].n_be; j++)
	      {
		printf("Unsorted:\t%s\tsorted: %s\n",backends[j],backends[sorted[j]]);
	      }
	  //transform the backendids so they match alphabetical order
	  for (j = 0; j < pulsars[i].N; j++)
	    {
	      int new = 0;
	      while (sorted[new] != pulsars[i].backends[j])
		new++;
	      if (verbose == 3)
		printf("%d\t%d\n",new,pulsars[i].backends[j]);
	      pulsars[i].backends[j] = new;
	    }
	  

	  free(sorted);

	  int ma = psr.nParam;

	  pulsars[i].N_m = pulsars[i].N - ma;
	  pulsars[i].G = my_matrix_alloc(pulsars[i].N,pulsars[i].N_m);
	  
	  //Maybe i need to reobtain the desing matrix all the time, maybe not
	  get_G_matrix(pulsars[i].G, psr, pulsars[i]);

	  *Ndim += pulsars[i].N_m;
	  *Ntot += pulsars[i].N;
	  pulsars[i].CWN = my_matrix_alloc(pulsars[i].N,pulsars[i].N);
      //pulsars[i].CWN = gsl_matrix_calloc(pulsars[i].N,pulsars[i].N); //initialize with 0s

#ifdef DM
	  pulsars[i].phi_inv = my_matrix_alloc(2*NFFT,2*NFFT);
	  pulsars[i].F = my_matrix_alloc(pulsars[i].N,2*NFFT);
	  pulsars[i].GF = my_matrix_alloc(pulsars[i].N_m,2*NFFT);
#else
	  pulsars[i].phi_inv = my_matrix_alloc(NFFT,NFFT);
	  pulsars[i].F = my_matrix_alloc(pulsars[i].N,NFFT);
	  pulsars[i].GF = my_matrix_alloc(pulsars[i].N_m,NFFT);
#endif
	  
	  pulsars[i].GNGinv =  my_matrix_alloc(pulsars[i].N_m,pulsars[i].N_m);
	  pulsars[i].H = my_matrix_alloc(pulsars[i].N,2);
	  
	  pulsars[i].C = my_matrix_alloc(pulsars[i].N_m,pulsars[i].N_m);
	  pulsars[i].Cinv = my_matrix_alloc(pulsars[i].N_m,pulsars[i].N_m);
	  pulsars[i].L = my_matrix_alloc(pulsars[i].N_m,pulsars[i].N_m);
	  pulsars[i].GH = my_matrix_alloc(pulsars[i].N_m,2);
	  
	  pulsars[i].Gres = my_vector_alloc(pulsars[i].N_m);
	} //end if only_res == 0
      //Do this part always
      my_dgemv(CblasTrans,1.0,pulsars[i].G,pulsars[i].res,0.0,pulsars[i].Gres);
      //      my_vector_print(pulsars[i].Gres);
      if (only_res == 0)
	{
	  compute_H(&(pulsars[i]),par);

#ifdef REDNOISE
      //read in sampled data
	  int xdim = 4+2*pulsars[i].n_be+1;
	  pulsars[i].sample = my_matrix_alloc(xdim,N_SAMPLE_MAX);
	  char infilename[100];
	  sprintf(infilename,"%s/chains_Noise/Noise_post_equal_weights.dat",pulsars[i].name);
	  if (verbose)
	    printf("Opening %s\n",infilename);
	  infile = fopen(infilename,"r");
	  j = 0;
	  int dobreak = 0;
	  while (1)
	    {
	      //      for (j = 0; j < pulsars[i].n_sample; j++)
	      for (k = 0; k < xdim; k++)
		if (fscanf(infile,"%le",&(pulsars[i].sample->data[j*xdim + k])) == EOF)
		  {
		    dobreak = 1;
		    break;
		  }
	      j++;
	      if (dobreak)
		break;
	    }
	  pulsars[i].n_sample = j;
	  if (verbose == 2)
	    {
	      printf("first sample line:\n");
	      for (k = 0; k < xdim; k++)
		printf("%g\t",pulsars[i].sample->data[k]);
	      printf("\n");
	    }
	  fclose(infile);
#endif
	  //      compute_Nwiggle(&(pulsars[i]));
	  double tmin= 1.0E10;
	  double tmax = 0.0;
      //  double temp;
	  int b;
	  for (b = 0; b < pulsars[i].N; b++)
	    {
	      if ((pulsars[i].toa->data[b]) > tmax)
		tmax = pulsars[i].toa->data[b];
	      if ((pulsars[i].toa->data[b]) < tmin)
		tmin = pulsars[i].toa->data[b];
	    }
	  temp = (tmax - tmin);
	  //temp = *tspan*1.306;                                                                                                                                                 
	  //  temp = *tspan;
	  //temp = *tspan*1.18;                                                                                                                                                  
	  double ffund = 1.0/(temp);
	  //set up frequency grid                                                                                                                                                
	  for (b = 0; b < (NFFT/2); b++)
	    {
	      pulsars[i].freqs[b] = ffund*(b+1);
	      //      printf("%d\t%e\n",i,par->freqs[i]);
	    }
	  pulsars[i].tspan = temp;
      
	}

    }

  if (verbose)
    printf("Ndim is %d\n",*Ndim);
}

//Get FFT matrix for one single pulsar given some DM for this backend
void initialize_fft_per_pulsar(struct mypulsar * psr, struct parameters * par)
{
  int i,b;
  double f;
  for (b = 0; b < psr->N; b++)
    {
      for (i = 0; i < (NFFT/2); i++)
	{
	  //	      f = pow(10.0,(logfmax - logfmin) * (float) i / (float) (NFFT/2.0) + logfmin);
	  //	  if (verbose > 2)
	  f = psr->freqs[i];
	  int row = i*2;
	  int row_cos = i*2+1;
	  psr->F->data[(row)*psr->F->m + b] = sin(2.0*PI*psr->toa->data[b]*f);///par->tspan;
	  psr->F->data[(row_cos)*psr->F->m + b] = cos(2.0*PI*psr->toa->data[b]*f);///par->tspan;
#ifdef DM
	  psr->F->data[(row+NFFT)*psr->F->m + b] = 1.0/(K_DM*psr->obsfreqs->data[b]*psr->obsfreqs->data[b])*sin(2.0*PI*psr->toa->data[b]*f);///par->tspan;
	  psr->F->data[(row_cos+NFFT)*psr->F->m +  b] = 1.0/(K_DM*psr->obsfreqs->data[b]*psr->obsfreqs->data[b])*cos(2.0*PI*psr->toa->data[b]*f);///par->tspan;
#endif	  
	}
    }
  //  my_matrix_print("F\0",psr->F);
  my_dgemm(CblasTrans,CblasNoTrans,1.0,psr->G,psr->F,0.0,psr->GF);
}

void calculate_phi_inv_per_pulsar(struct mypulsar * psr,struct parameters par)
{
  int i,a,b;
  double f;
  double power;
  //Red noise will change all the time, lets give it as two parameters
  //phi for one pulsar depends only on red noise

  //  double rfac = rA * rA / (12.0*PI*PI) * 3.16e22;
  double rAgw = psr->rA;
  double rgamma = psr->rgamma;
  double rfac = rAgw * rAgw * 3.16e22 / (12.0*PI*PI) ;
#ifdef DM
  double dmAgw = psr->dmA;
  double dmgamma = psr->dmgamma;
  double dmfac = dmAgw * dmAgw * 3.16e22;// / (12.0*PI*PI) ;
#endif
  for (i = 0; i < NFFT/2; i++)
    {
      f = psr->freqs[i];
      //      power = rfac* pow(f/3.17e-08,-rgamma) / par.tspan;
      //add red noise for diagonal elements
      //      power = rfac * pow(f/3.17e-08,-rgamma)/psr->tspan;
      power = rfac * pow(f/3.17e-08,-rgamma)/psr->tspan;
      psr->phi_inv->data[(2*i)*psr->phi_inv->m  + 2*i] = 1.0/power;
      psr->phi_inv->data[(2*i+1)*psr->phi_inv->m + 2*i + 1] = 1.0/power;
#ifdef DM
      //      power = dmfac * pow(f/3.17e-08,-dmgamma)/psr->tspan;
      power = dmfac * pow(f/3.17e-08,-dmgamma)/psr->tspan;
      psr->phi_inv->data[(NFFT+2*i)*psr->phi_inv->m  + NFFT + 2*i] = 1.0/power;
      psr->phi_inv->data[(NFFT+2*i+1)*psr->phi_inv->m + NFFT + 2*i + 1] = 1.0/power;
#endif
    }
}

void compute_C_matrix(struct mypulsar * psr, struct parameters * par)
{
  if (verbose ==2)
    printf("Compute Nwiggle\n");
  double tstart = omp_get_wtime();

  compute_Nwiggle(psr); //this will also obtain GNGinv and set new noise parameters
  double tend = omp_get_wtime();
  if (verbose == 3)
    printf("compute_Nwiggle\t\t\t%g\n",tend-tstart);
  //rebuild fft, DM might have changed
  if (verbose ==2)
    printf("Compute fft\n");
  tstart = omp_get_wtime();
  initialize_fft_per_pulsar(psr,par);
  tend = omp_get_wtime();
  if (verbose == 3)
    printf("compute_fft\t\t\t%g\n",tend-tstart);
  //biuld phi_inv directly, only diagonal
  if (verbose ==2)
    printf("Compute phi\n");
  tstart = omp_get_wtime();
  calculate_phi_inv_per_pulsar(psr,*par);
  tend = omp_get_wtime();
  if (verbose == 3)
    printf("compute_phi\t\t\t%g\n",tend-tstart);
  //	  my_matrix_print("phi inv\0",psr->phi_inv);
  //build up F N F
  if (verbose ==2)
    printf("Compute FNF\n");
  struct my_matrix * FNF, *FN;
#ifdef DM
  FN = my_matrix_alloc(2*NFFT,psr->N_m);
  FNF = my_matrix_alloc(2*NFFT,2*NFFT);
#else
  FN = my_matrix_alloc(NFFT,psr->N_m);
  FNF = my_matrix_alloc(NFFT,NFFT);
#endif
  my_dgemm(CblasTrans,CblasNoTrans,1.0,psr->GF,psr->GNGinv,0.0,FN);
  my_dgemm(CblasNoTrans,CblasNoTrans,1.0,FN,psr->GF,0.0,FNF);
  // end FNF
  
  //add to phi_inv and invert
  
  if (verbose ==2)
    printf("invert FNF plus phi\n");
  struct my_matrix * FNF_lu;
#ifdef DM
  FNF_lu = my_matrix_alloc(2*NFFT,2*NFFT);
#else
  FNF_lu = my_matrix_alloc(NFFT,NFFT);
#endif

  my_matrix_add(FNF,psr->phi_inv);
  
  double detFNF;
#ifdef DM
  get_inverse_lu(FNF,FNF_lu,2*NFFT,&detFNF);
#else
  get_inverse_lu(FNF,FNF_lu,NFFT,&detFNF);
#endif
  
  //reuse F*N
  struct my_matrix *FNFFN;
#ifdef DM
  FNFFN = my_matrix_alloc(2*NFFT,psr->N_m);
#else
  FNFFN = my_matrix_alloc(NFFT,psr->N_m);
#endif
  my_dgemm(CblasNoTrans,CblasNoTrans,1.0,FNF,FN,0.0,FNFFN);
  my_dgemm(CblasTrans,CblasNoTrans,1.0,FN,FNFFN,0.0,psr->Cinv);
  //my_matrix_print("Nwiggle\0",pulsars[i].Nwiggle);
  //      my_matrix_print("C\0",C);
#ifndef CUNITY  
#ifdef REDNOISE
  my_matrix_sub(psr->GNGinv,psr->Cinv);
#else
  my_matrix_memcpy(psr->Cinv,psr->GNGinv);
#endif
#else
  my_matrix_set_zero(psr->Cinv);
  int i;
  for (i = 0; i < psr->N_m; i++)
    {
      psr->Cinv->data[psr->N_m*i + i] = 1.0;
    }
#endif
      //now C is C^-1!
  my_matrix_free(FNFFN);
  my_matrix_free(FNF_lu);
  my_matrix_free(FN);
  my_matrix_free(FNF);

}

#ifdef UPPER
struct geo_par ComputeFs(double thetaS, double phiS, double theta_a, double phi_a )
{
  struct geo_par temp;
  double  sinX = sin(theta_a);
  double  cosX = cos(theta_a);
  double  sinX2 = sinX*sinX;
  double  cosX2 = cosX*cosX;
  double  cthS = cos(thetaS);
  double  sthS = sin(thetaS);
  double  dphig = phiS - phi_a;
  double  kn = -sthS*sinX*cos(dphig) - cthS*cosX;

  temp.Fac = (-0.25*(sinX2 - 2.*cosX2)*sthS*sthS - cosX*sinX*sthS*cthS*cos(dphig) +  0.25*(1. + cthS*cthS)*sinX2*cos(2.*dphig))/(1. + kn);
    
  temp.Fas = (cosX*sinX*sthS*sin(dphig) - 0.5*sinX2*cthS*sin(2.*dphig))/(1.+ kn);

  return temp;

}

void randomize_source(struct source * p, gsl_rng * r)
{
  double McRange[2] = {1.e8,1.e10};
  double beta = -2./3.;
  double normMc = ( pow(McRange[1],(beta+1.0)) - pow(McRange[0],(beta+1.0)))/(beta+1.0);
  double AA = gsl_rng_uniform(r)*normMc;
  double Mc = pow(( pow(McRange[0],(beta+1.0)) + (beta+1.0)*AA ),(1.0/(beta + 1.0)));
  double DL = 2.0*pow((Mc*Msun),(5./3.)) *pow((PI*p->fr),(2./3.))/p->Amp;
  double th_s = acos(2.0*(gsl_rng_uniform(r)-0.5));
  double ph_s = gsl_rng_uniform(r)*2.0*PI;
  double psi = gsl_rng_uniform(r)*PI;
  double phi0 = gsl_rng_uniform(r)*2.0*PI;
  double incl = acos(2.0*(gsl_rng_uniform(r)-0.5));

  p->theta_s = th_s;
  p->phi_s = ph_s;
  p->Mc = Mc;
  p->psi = psi;
  p->phi0 = phi0;
  //p->Amp = 1e-15;
  p->iota = incl;
  
  //  p->fr = 2.0e-7;
}

void add_signal(struct mypulsar *psr, pulsar * t_psr, struct parameters params, struct source source_pars)
{
  double theta_a = 0.5*PI - psr->dec;
  double phi_a = psr->raj;
  double L_a = psr->dist; //distance to the pulsar
  
  double theta_s = source_pars.theta_s;
  double phi_s = source_pars.phi_s;
  double Mc = source_pars.Mc;
  double fr = source_pars.fr;
  //double fr = params.omega/(2.0*PI);
  
  double psi = source_pars.psi;
  double phi0 = source_pars.phi0;
  double Amp = source_pars.Amp;
  double iota = source_pars.iota;

  double om_0 = 2.*PI*fr;
   
  double a1 = Amp*( (1.0 + cos(iota)*cos(iota))*cos(psi)*cos(phi0) - 2.0*cos(iota)*sin(psi)*sin(phi0) );
  double a2 = Amp*( (1.0 + cos(iota)*cos(iota))*sin(psi)*cos(phi0) + 2.0*cos(iota)*cos(psi)*sin(phi0) );
  double a3 = Amp*( (1.0 + cos(iota)*cos(iota))*cos(psi)*sin(phi0) + 2.0*cos(iota)*sin(psi)*cos(phi0) );
  double a4 = Amp*( (1.0 + cos(iota)*cos(iota))*sin(psi)*sin(phi0) - 2.0*cos(iota)*cos(psi)*cos(phi0) );
   
  double k[3] = {sin(theta_s)*cos(phi_s), sin(theta_s)*sin(phi_s), cos(theta_s)};
  double n[3] = {sin(theta_a)*cos(phi_a), sin(theta_a)*sin(phi_a), cos(theta_a)};
  double Lp = L_a*1.e3*pc_sec;
  double Mc_sec = Mc*Msun;
  
  double tau = Lp*(1. + k[0]*n[0] + k[1]*n[1] + k[2]*n[2]);
  double om_orb = 0.5*om_0;
  double om_p =  om_0*pow((1.0 + 256./5.*pow(Mc_sec,(5./3.))*pow(om_orb,(8./3.))*tau),(-3./8.)) ;
  double phi_a_phase = -om_0*tau;

  struct geo_par geo;
  geo = ComputeFs(theta_s, phi_s, theta_a, phi_a);

  int i;
  double phase_e,phase_p,sn,cs,sn_p,cs_p,h;
  double denom_2 = (pow(om_p,(1./3.))*pow(om_0,(2./3.)));
  for (i = 0; i < t_psr->nobs; i++)
    {
      phase_e = om_0*psr->oldbat[i];
      phase_p = om_p*psr->oldbat[i] + phi_a_phase;

      sn = sin(phase_e)/(om_0);
      cs = cos(phase_e)/(om_0);
      sn_p = sin(phase_p)/denom_2;
      cs_p = cos(phase_p)/denom_2;

     
      //add signal to the residual
      t_psr->obsn[i].sat = (psr->oldbat[i] + (a1*geo.Fac + a2*geo.Fas)*(sn_p - sn) + (a3*geo.Fac + a4*geo.Fas)*(cs_p -cs))/86400.0;
    }

}
#endif

//#ifdef SINGLE
//double compute_likelihood(struct my_matrix * phi,struct my_matrix * FNF, struct my_vector * FNT,double tNt, double detN, int Nplsr, double ss, double detF)
//#else
//double compute_likelihood(struct my_matrix * phi,struct my_matrix * FNF, struct my_vector * FNT,double tNt, struct my_matrix * Ninv, double detN, int Nplsr)
//#endif
void create_noise_only(struct mypulsar * psr, gsl_rng * r)
{
  //draw from multivariate gaussian by inverting C
  //first invert C^-1
  int i;

//  struct my_vector * w;
//  w = my_vector_alloc(psr->N_m);
  for (i = 0; i < psr->N_m; i++)
    //    psr->Gres->data[i] = gsl_ran_gaussian(r,1.0);
    psr->Gres->data[i] = gsl_ran_gaussian(r,1.0);

  if (verbose)
    printf("Creating noise\n");
  struct my_matrix *Linv;
  Linv = my_matrix_alloc(psr->N_m,psr->N_m);
  my_matrix_memcpy(psr->C,psr->Cinv);
  //  my_symcheck(psr->C);
  //call it twice to have in the end Linv the L of the C (stupid but working)
  get_inverse_cholesky(psr->C,Linv,psr->N_m);
  //  my_matrix_print("L\0",Linv);
  get_inverse_cholesky(psr->C,psr->L,psr->N_m);
  
//
//
//
//  my_matrix_memcpy(psr->L,Linv);
//
//  get_inverse_tri(psr->L);
//  //now this L is what I need to form the multivariate gaussian, I hope
  //create unit random vector w
  //multiply it into residuals
  //  my_matrix_print("L\0",psr->L);
//  int j;
//
//  for (i = 0; i < psr->L->m; i++)
//    for (j = 0; j < i; j++)
//      {
//	psr->L->data[i*psr->L->m + j] = 0.0;
//	psr->C->data[i*psr->C->m + j] = psr->C->data[j*psr->C->m + i];
//      }

  my_dtrmv(CblasNoTrans,psr->L,psr->Gres);
  //set upper diagonal zero
//  my_matrix_memcpy(psr->L,Linv);
//  //multiply L Lt and see if it is really C
//  struct my_matrix * Ctemp;
//  Ctemp = my_matrix_alloc(psr->N_m,psr->N_m);
//  my_dgemm(CblasNoTrans,CblasTrans,1.0,psr->L,Linv,0.0,Ctemp);
//  my_matrix_print("Ctemp\0",Ctemp);
//  my_matrix_free(Ctemp);
  //  printf("only noise\n");
  //  my_vector_print(psr->Gres);
  //done?
  my_matrix_free(Linv);
}

struct Fp compute_Fp(struct mypulsar * pulsars, struct parameters * par, int Nplsr)
{
  if (verbose == 1)
    printf("Entering compute_Fp\n");
  struct Fp coeff;
  coeff.tCt = 0.0;
  coeff.tHt = 0.0;
  int i;
  int used = 0;
  //build matrices
  for (i = 0; i < Nplsr; i++)
    {
      //use only pulsars with appropriate observation span
      if ((1.0/pulsars[i].tspan) > (par->omega/(2.0*PI)))
	{
	  if (verbose)
	    {
	      printf("Skipping %s\t for freq %e, 1/tspan is \t%e\n",pulsars[i].name,par->omega/(2.0*PI),1.0/pulsars[i].tspan);
	    }
	  continue;
	}
      used++;
      //get inner product
      struct my_vector *Ct;
      Ct = my_vector_alloc(pulsars[i].N_m);
      my_dgemv(CblasNoTrans,1.0,pulsars[i].Cinv,pulsars[i].Gres,0.0,Ct);
      double tCt = 0.0;
      my_vector_mult(pulsars[i].Gres,Ct,&tCt);

      if (verbose == 2)
	printf("tCt %s %d\t%g\n",pulsars[i].name,i,tCt);
      coeff.tCt += 0.5*tCt;

      //build up the second term: tH HH Ht
      //recompute H for frequency
      compute_H(&(pulsars[i]),par);

      //need matric H^T C H
      struct my_matrix * CH;
      CH = my_matrix_alloc(pulsars[i].N_m,2);
      struct my_matrix * HCH;
      HCH = my_matrix_alloc(2,2);
      my_dgemm(CblasNoTrans,CblasNoTrans,1.0,pulsars[i].Cinv,pulsars[i].GH,0.0,CH);
      my_dgemm(CblasTrans,CblasNoTrans,1.0,pulsars[i].GH,CH,0.0,HCH);
      //invert
      //      my_matrix_print("H\0",pulsars[i].H);
      struct my_matrix * HCH_lu;
      HCH_lu = my_matrix_alloc(2,2);
      double detHCH;
      get_inverse_lu(HCH,HCH_lu,2,&detHCH);

      //obtain the 2 dim vector Ht
      struct my_vector * Ht;
      Ht = my_vector_alloc(2);

      my_dgemv(CblasTrans,1.0,CH,pulsars[i].Gres,0.0,Ht);

      struct my_vector * HHtH;
      HHtH = my_vector_alloc(2);
      my_dgemv(CblasNoTrans,1.0,HCH,Ht,0.0,HHtH);
      double tHt = 0.0;
      my_vector_mult(Ht,HHtH,&tHt);

      if (verbose == 2)
	printf("tHt %s %d\t%g\n",pulsars[i].name,i,tHt);
      coeff.tHt += 0.5*tHt;
      
      my_vector_free(Ht);
      my_vector_free(HHtH);
      my_matrix_free(HCH);
      my_matrix_free(CH);
      my_matrix_free(HCH_lu);
      my_vector_free(Ct);
    }
  coeff.used = used;
  return coeff;
}

////initialize_FNF(GT,FNF,FNT,G_large,GNG_large,&tNt,pulsars,F,S,Ntot,Nplsr,&detGNG,Ndim); 
//
//void initialize_FNF(struct my_vector * GT, struct my_matrix *G_large, struct my_matrix * GNG_large, struct mypulsar * pulsars, struct my_matrix * F,struct my_matrix *S, int Ntot, int Nplsr, int Ndim)
//{
//  int a,b,c,ind_c,ind_r;
//  ind_r = 0; ind_c = 0;
//  for (a = 0; a < Nplsr; a++)
//    {
//      for (b = 0; b < pulsars[a].N_m; b++)
//	{
//	  for (c = 0; c < pulsars[a].N_m; c++)
//	    {
//	      GNG_large->data[(ind_c+c)*GNG_large->m + ind_r + b] = pulsars[a].GNG->data[c*pulsars[a].GNG->m+b];
//	    }
//	}
//      ind_r += pulsars[a].N_m;
//      ind_c += pulsars[a].N_m;
//    }
//  //  my_matrix_print("GNG LARGE\0",GNG_large);
//  ind_r = 0; ind_c = 0;
//  for (a = 0; a < Nplsr; a++)
//    {
//      for (b = 0; b < pulsars[a].N; b++)
//	{
//	  for (c = 0; c < pulsars[a].N_m; c++)
//	    {
//	      G_large->data[(ind_c+b)*G_large->m + ind_r + c] = pulsars[a].G->data[c*pulsars[a].G->m+b];
//	    }
//	}
//      ind_r += pulsars[a].N_m;
//      ind_c += pulsars[a].N;
//    }
//  //  my_matrix_print("GLARGE\0",G_large);
//  //#ifndef SINGLE
//  if (verbose)
//    printf("Multiplying G * F\n");
//  my_dgemm(CblasNoTrans,CblasNoTrans,1.0,G_large,F,0.0,S);
//  //make one large vector for all the residuals
//  struct my_vector * TOAlarge;
//  TOAlarge = my_vector_alloc(Ntot);
//  ind_c = 0;
//  for (a = 0; a < Nplsr; a++)
//    {
//      for (b = 0; b < pulsars[a].N; b++)
//	{
//	  TOAlarge->data[ind_c+b] = pulsars[a].res[b];
//	}
//      ind_c += pulsars[a].N;
//    }
//  //build GT
//  my_dgemv(CblasNoTrans,1.0,G_large,TOAlarge,0.0,GT);
//  //print GT
//  //  my_vector_print(GT);
//  my_vector_free(TOAlarge);
//  //#endif
//
//}

#if defined(MCMC) || defined(GRID)
#ifndef MPI //this is a mcmc with random starting points
void propose_jump(struct parameters * new, struct parameters old, gsl_rng *r, double etadyn)
{
  int i;
  for (i = 0; i < NCOEFF; i++)
    {
      new->values[i] = old.values[i] + gsl_ran_gaussian(r,etadyn*(old.u[i] - old.l[i]));
      //check boundaries
      if (new->values[i] < new->l[i])
	new->values[i] = new->l[i];
      if (new->values[i] > new->u[i])
	new->values[i] = new->u[i];
    }
  
}

void copyparams(struct parameters *new, struct parameters old)
{
  int i;
  for (i = 0; i < NCOEFF; i++)
    {
      new->values[i] = old.values[i];
    }
}
#endif //ndef MPI
#endif //MCMC

/*
  MAIN ****************************************
 */

int main(int argc, char *argv[])
{
#ifdef MPI
  //  double tstart = omp_get_wtime();
  int rank, numproc;
  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//  if (rank == 0)
//    {
#endif

#ifdef CULA
  culaStatus s;
#ifdef MPI
//  if (numproc > 4)
//    fprintf(stderr,"This is not safe on damiana! We only have 4 gpu devices!\n");
  culaSelectDevice(rank);
//#else
//  culaSelectDevice(0);
#endif
#ifdef UPPER
  culaSelectDevice(1);
#endif
s = culaInitialize();
  if(s != culaNoError)
    {
      printf("%s\n", culaGetStatusString(s));
      /* ... Error Handling ... */
    }
#endif

  srand(time(NULL));

  int Nplsr;//,N,N_m;
  int i,j,k;
  //  char **filenames;
  char filenames[MAX_PSR][MAX_FILELEN];
  char timfilenames[MAX_PSR][MAX_FILELEN];
  char parfilenames[MAX_PSR][MAX_FILELEN];
  char pulsarname[MAX_PSR][MAX_FILELEN];
  int Nrows,Ndim,Ntot;
  FILE *ofile;

  //to hold all pulsar information, including individual G matrices
  struct mypulsar *pulsars;
  pulsar * tempo_psrs;
  ofile = fopen(argv[1],"w");
#ifdef UPPER
  Nplsr = (argc-4); //because 2nd and 3rd argument are frequency, amplitude and detection threshold
#else
  Nplsr = (argc-2);
#endif
  //  filenames = (char **) malloc(Nplsr*sizeof(char *));

//  timfilenames = (char **) malloc(Nplsr*sizeof(char *));
//  parfilenames = (char **) malloc(Nplsr*sizeof(char *));
  //read filenames
  
  tempo_psrs = (pulsar *) malloc(MAX_PSR*sizeof(pulsar));

  struct parameters params;
  params.omega = 1e-8;

  for (i = 0; i < Nplsr; i++)
    {
      //      filenames[i] = (char *) malloc(60*sizeof(char));
#ifdef UPPER
      strcpy(pulsarname[i],argv[i+4]);
      sprintf(filenames[i],"%s/%s_mod.tim",pulsarname[i],pulsarname[i]);
      sprintf(parfilenames[i],"%s/%s.par",pulsarname[i],pulsarname[i]);
#else
//      strcpy(filenames[i],argv[i+2]);
//      strcpy(parfilenames[i],argv[i+2+Nplsr]);
      strcpy(pulsarname[i],argv[i+2]);
      sprintf(filenames[i],"%s/%s_mod.tim",pulsarname[i],pulsarname[i]);
      sprintf(parfilenames[i],"%s/%s.par",pulsarname[i],pulsarname[i]);
#endif
      if (verbose)
	printf("Read %s\t%s\n",filenames[i],parfilenames[i]);
    }
  initialise(tempo_psrs,0);
  readParfile(tempo_psrs,parfilenames,filenames,Nplsr);
  readTimfile(tempo_psrs,filenames,Nplsr);
  preProcess(tempo_psrs,Nplsr,argc,argv);
  formBatsAll(tempo_psrs,Nplsr);
  formResiduals(tempo_psrs,Nplsr,0.0);
  double globalParameter;
  double tstart = omp_get_wtime();
  
  doFitAll(tempo_psrs,Nplsr,0);

  double tend = omp_get_wtime();
  printf("%g\n",tend-tstart);
//  for (i = 0; i < tempo_psrs[0].nobs; i++)
//    {
//      printf("res\t%lg\n",(double) tempo_psrs[0].obsn[i].residual);
//    }
//
  pulsars = (struct mypulsar *) malloc(Nplsr * sizeof(struct mypulsar));

  for (i = 0; i < Nplsr; i++)
    {
      strcpy(pulsars[i].name,pulsarname[i]);
      //find distance from distance.dat file
      int foundit = 0;
      for (j = 0; j < HAVEDIST; j++)
	{
	  if (strcmp(pulsarname[i],distance_keys[j]) == 0)
	    {
	      pulsars[i].dist = distance_values[j];
	      foundit = 1;
	      //	      printf("Found distance\t%s\t%g\n",pulsarname[i],pulsars[i].dist);
	    }
	}
      if (foundit == 0)
	pulsars[i].dist = 1.0;

    }

  gsl_rng *r;
  r = gsl_rng_alloc (gsl_rng_mt19937);
  gsl_rng_set(r,time(NULL));
  /* ------------------------- */
  tstart = omp_get_wtime();
  initialize_pulsars_fromtempo(tempo_psrs,pulsars,Nplsr,&Ndim,&Ntot,&params,0);
  tend = omp_get_wtime();
  //  printf("init took %f sec\n",tend-tstart);

  //Ntot = total number of residuals.pu Ndim = reduced number of residuals
  /* ------------------------- */
  int changed = 1;

  params.omega = 3.0e-7;
  for (i = 0; i < Nplsr; i++)
    {
#ifdef REDNOISE // values for data challenge 3 open
      pulsars[i].rA = pow(10.0,-14.35);
      pulsars[i].rgamma = 1.52;
#endif
      pulsars[i].index = pulsars[i].n_sample-NF0-5;
      //      pulsars[i].index = 0;
      compute_C_matrix(&(pulsars[i]),&params);
    }


  FILE * outfile;
  char oname[50];
  double tstart_tot = omp_get_wtime();

  struct Fp Fp;

  Fp = compute_Fp(pulsars,&params,Nplsr);
  printf("#\t% 6.4e% 6.4e\n",Fp.tCt,Fp.tHt);
  //find longest tspan for min freq
  params.tspan = 0.0;
  for (i = 0; i < Nplsr; i++)
    if (pulsars[i].tspan > params.tspan)
      params.tspan = pulsars[i].tspan;
  //create array of frequencies to be investigated
#ifdef F0
  int nfreqs = 1;
  double fstep = 0.01;
  double fmin = log10(3.0e-7);
  double * h_freqs;
#else
  double fstep = 0.01;
  double fmin = log10(1.0/params.tspan);
  double fmax = log10(5.e-07);
  double * h_freqs;
  int nfreqs = (int) ((fmax-fmin)/fstep);
#endif
  h_freqs = (double *) malloc(nfreqs*sizeof(double));
  //fill freqs
  for (i = 0; i < nfreqs; i++)
    {
      h_freqs[i] = pow(10.0,fmin + i*fstep);
    }
  int ifreq;

#ifdef UPPER
  struct source source_pars;
  params.omega = 2.0*PI*atof(argv[2]);

  //put zero source and iterate 5 times
  source_pars.Amp = 0.0;
  for (i = 0; i < 10; i++)
    {
      source_pars.fr = params.omega/(2.0*PI);
      randomize_source(&source_pars,r);
      for (j = 0; j < Nplsr; j++)
	{
	  add_signal(&(pulsars[j]),&(tempo_psrs[j]),params,source_pars);
	  //pulsars[j].index = gsl_rng_uniform_int(r,pulsars[j].n_sample-1);;
	}

      formBatsAll(tempo_psrs,Nplsr);
      formResiduals(tempo_psrs,Nplsr,0.0);
      doFitAll(tempo_psrs,Nplsr,0);

      initialize_pulsars_fromtempo(tempo_psrs,pulsars,Nplsr,&Ndim,&Ntot,&params,1);

    }
  for (j = 0; j < Nplsr; j++)
    compute_C_matrix(&(pulsars[j]),&params);
  Fp = compute_Fp(pulsars,&params,Nplsr);

  source_pars.Amp = 0.4*pow(10.0,atof(argv[3])); // factor of sqrt(5)/sqrt(2)/4
  //compute to find out how many pulsars we are using here to get detection threshold value
  //  double threshs[45] = { 0.000000, 14.700000, 17.600000, 20.100000, 22.350000, 24.450000, 26.500000, 28.400000, 30.300000, 32.100000, 33.900000, 35.650000, 37.350000, 39.000000, 40.650000, 42.300000, 43.900000, 45.500000, 47.100000, 48.650000, 50.200000, 51.750000, 53.250000, 54.800000, 56.300000, 57.750000, 59.250000, 60.700000, 62.200000, 63.650000, 65.100000, 66.550000, 67.950000, 69.400000, 70.800000, 72.250000, 73.650000, 75.050000, 76.450000, 77.850000, 79.200000, 80.600000, 82.000000, 83.350000, 84.750000};
  double threshold = Fp.tHt;//threshs[Fp.used];
  if (verbose == 2)
    printf("Amplitude: %g\tThresh: %f\n",source_pars.Amp,threshold);
  //double threshold = atof(argv[4]);
  int detected = 0;
  int total = 0;
  for (i = 0; i < NF0; i++)
    {
      source_pars.fr = params.omega/(2.0*PI);
      randomize_source(&source_pars,r);
      for (j = 0; j < Nplsr; j++)
	{
	  add_signal(&(pulsars[j]),&(tempo_psrs[j]),params,source_pars);
	  pulsars[j].index = gsl_rng_uniform_int(r,pulsars[j].n_sample-1);;
	}

      formBatsAll(tempo_psrs,Nplsr);
      formResiduals(tempo_psrs,Nplsr,0.0);
      doFitAll(tempo_psrs,Nplsr,0);

      initialize_pulsars_fromtempo(tempo_psrs,pulsars,Nplsr,&Ndim,&Ntot,&params,1);

      for (j = 0; j < Nplsr; j++)
	compute_C_matrix(&(pulsars[j]),&params);
      Fp = compute_Fp(pulsars,&params,Nplsr);
      if (verbose)
	printf("%f\t%g\n",Fp.tHt,threshold);
      if (Fp.tHt > threshold)
	detected++;
      total++;
      tend = omp_get_wtime();
      if (verbose == 3)
	printf("compute_Fp \t%g\n",tend-tstart);
      //  fprintf(outfile,"% 6.4e\t% 6.4e\n",Fp.tCt,Fp.tHt);
      //printf("% 6.4e\n",Fp.tHt);
    }
  double fdetected = (double) detected / total;
  printf("DETECTED\t%f\n",fdetected);

#else


#ifndef MPI
  //  for (params.omega = 7.*(2.0*PI); params.omega < (1e-07*(2.0*PI)); params.omega += (1e-8*2.0*PI))
  for (ifreq = 0; ifreq < nfreqs; ifreq++)
    {
      params.omega = h_freqs[ifreq]*2.0*PI;
#else

  int n_per_proc = nfreqs/numproc;
  printf("Running %d per proc, total %d procs and %d freqs\n",n_per_proc,numproc,nfreqs);
  for (ifreq = 0; ifreq < n_per_proc; ifreq++)
    {
      params.omega = h_freqs[rank*n_per_proc + ifreq] *2.0*PI;
#endif
#ifdef F0
      sprintf(oname,"%6.4e.fp",params.omega/(2.0*PI));
      outfile = fopen(oname,"w");
      //      pulsars[0].index = 23;
      for (i = 0; i < NF0; i++)
	{
	  //fill up psrs with noise
	  //      for (j = 0; j < Nplsr; j++)
	  double tstart = omp_get_wtime();

	  for (j = 0; j < Nplsr; j++)
	    {
	      //	      pulsars[j].index += 2;
	      pulsars[j].index = gsl_rng_uniform_int(r,pulsars[j].n_sample-1);
	      compute_C_matrix(&(pulsars[j]),&params);
	    }

	  double tend = omp_get_wtime();
	  if (verbose == 3)
	    printf("compute_C_matrix\t%g\n",tend-tstart);
	  //	  compute_Nwiggle(&(pulsars[0]));
	  tstart = omp_get_wtime();

    //    psr->Gres->data[i] = gsl_ran_gaussian(r,1.0);
	  
	  for (j = 0; j < Nplsr; j++)
	    create_noise_only(&(pulsars[j]),r);
	  //Do i want to create a new C for analyzing the things?
	  for (j = 0; j < Nplsr; j++)
	    {
	      //	      pulsars[j].index += 2;
	      pulsars[j].index = gsl_rng_uniform_int(r,pulsars[j].n_sample-1);
	      compute_C_matrix(&(pulsars[j]),&params);
	    }


	  tend = omp_get_wtime();
	  if (verbose == 3)
	    printf("create_noise \t%g\n",tend-tstart);
	  tstart = omp_get_wtime();
	  Fp = compute_Fp(pulsars,&params,Nplsr);
	  tend = omp_get_wtime();
	  if (verbose == 3)
	    printf("compute_Fp \t%g\n",tend-tstart);
	  fprintf(outfile,"% 6.4e\t% 6.4e\t%d\n",Fp.tCt,Fp.tHt,Fp.used);
	  fflush(outfile);
	  //printf("% 6.4e\n",Fp.tHt);
	}
//      double tend_tot = omp_get_wtime();
//      if (verbose == 3)
//	pri<ntf("Duration all evaluation:\t%g\n",tend_tot-tstart_tot);
      fclose(outfile);
#else
      Fp = compute_Fp(pulsars,&params,Nplsr);
      changed = 0;
      fprintf(ofile,"% 6.4e  % 6.4e  % 6.4e %d\n",params.omega,Fp.tCt,Fp.tHt,Fp.used);
#endif
    }
  double tend_tot = omp_get_wtime();
  fprintf(stderr,"Duration all evaluation:\t%g\n",tend_tot-tstart_tot);
#endif //else UPPER
#ifdef CULA
  culaShutdown();
#endif
#ifdef MPI
  if(rank ==0)
#endif
    fclose(ofile);
#ifdef MPI
  MPI_Finalize();
#endif
  return 0;
}
