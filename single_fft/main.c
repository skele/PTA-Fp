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
#define MAX_BE 20

//#define ANGLES

#include <mydefs.h>

#include <twalk.h>
//#define UPPER
//#define EFAC
#define DM

//#define GWB

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
//#define RUNITY
//#define SS
#define REDNOISE
#define GLOBAL
#define POWERLAW
//#define SWITCHEROO

#define ETA 0.05

void compute_Nwiggle(struct mypulsar * pulsar)
{
  int N = pulsar->N;
  int N_m = pulsar->N_m;
  struct my_matrix * GNG, *GNG_temp, *Cholesky;
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
#ifdef POWERLAW
#ifndef GLOBAL
#ifdef REDNOISE
  pulsar->rA = pow(10.0,pulsar->sample->data[pulsar->index*pulsar->sample->m + 0]);
  pulsar->rgamma = pulsar->sample->data[pulsar->index*pulsar->sample->m + 1];
#endif
#ifdef DM
  pulsar->dmA = pow(10.0,pulsar->sample->data[pulsar->index*pulsar->sample->m + 2]);
  pulsar->dmgamma = pulsar->sample->data[pulsar->index*pulsar->sample->m + 3];
  if (verbose == 2)
    printf("rednoise %g\t%g\t DM    %g\t%g\n",pulsar->rA,pulsar->rgamma,pulsar->dmA,pulsar->dmgamma);
#endif
#endif

#endif
  for (j = 0; j < pulsar->N; j++)
    for (k = 0; k < pulsar->N; k++)
      {
	if (j == k)
	  {
#ifdef EFAC
	    ibe = pulsar->backends[j];
	    efac = pulsar->efac[ibe];
	    equad = pow(10.0,pulsar->equad[ibe]);
	    //	    printf("efac %d %g\t%g\n",ibe,efac,equad);
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
  pulsar->det = det;

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

void init_a_ab(struct mypulsar *pulsars, struct my_matrix* a_ab, int Nplsr)
{
  int i,j;
  double raj_a,raj_b,dec_a,dec_b,cth;
  for (i = 0; i < Nplsr; i++)
    {
      raj_a = pulsars[i].raj;
      dec_a = pulsars[i].dec;
      for (j = 0; j < Nplsr; j++)
        {
          if (i == j)
            {
              a_ab->data[i*Nplsr+j] = 1.0;
              //              gsl_matrix_set(a_ab,i,j,1.0);                                                                                                              
              continue;
            }
          else
            {
              raj_b = pulsars[j].raj;
              dec_b = pulsars[j].dec;
              cth = cos(dec_a)*cos(dec_b)*cos(raj_a-raj_b) + sin(dec_a)*sin(dec_b);
              cth = 0.5*(1.0-cth);
              a_ab->data[j*Nplsr+i] = 1.5*cth*log(cth) - 0.25*cth + 0.5;
              //              gsl_matrix_set(a_ab,i,j,1.5*cth*log(cth) - 0.25*cth + 0.5);                                                                                
	      //              gsl_matrix_set(a_ab,i,j,0.0);                                                                                                              
            }
        }
    }
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

  my_vector_free(singulars);
    my_matrix_free(design);
    my_matrix_free(U);
    my_matrix_free(designT);

}

void initialize_pulsars_fromtempo(pulsar * tempo_psrs, struct mypulsar * pulsars, int Nplsr, int *Ndim, int *Ntot, struct parameters * par, int *alldim, int only_res)
{
  FILE *infile;
  int i,j,k,l;
  if (verbose)
    printf("initializing from tempo\n");
  double temp;
  //initialize a struct for each pulsar
  *Ndim = 0;
  *Ntot = 0;
#ifdef GWB
  int totstride = NFFT/2;//for first pulsar, because before we put GWB amplitudes
#else
  int totstride = 0;
#endif
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
	  //allocate efacs and equads
	  pulsars[i].efac = (double *) malloc(pulsars[i].n_be * sizeof(double));
	  pulsars[i].equad = (double *) malloc(pulsars[i].n_be * sizeof(double));

	  //now I can compute the stride, i.e. the total amount of parameters for this pulsar to give to the next pulsar
	  pulsars[i].stride = totstride;
	  if (verbose)
	    printf("Stride is %d\t%d\n",i,totstride);
	  totstride += NFFT + 2*pulsars[i].n_be; //RN + DM + efac + equad
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
	}
	  //Maybe i need to reobtain the desing matrix all the time, maybe not
      get_G_matrix(pulsars[i].G, psr, pulsars[i]);
      if (only_res == 0)
	{
	  *Ndim += pulsars[i].N_m;
	  *Ntot += pulsars[i].N;
	  pulsars[i].CWN = my_matrix_alloc(pulsars[i].N,pulsars[i].N);
      //pulsars[i].CWN = gsl_matrix_calloc(pulsars[i].N,pulsars[i].N); //initialize with 0s

#ifdef DM
	  pulsars[i].phi_inv = my_matrix_alloc(2*NFFT,2*NFFT);
	  pulsars[i].F = my_matrix_alloc(pulsars[i].N,2*NFFT);
	  pulsars[i].GF = my_matrix_alloc(pulsars[i].N_m,2*NFFT);
	  pulsars[i].FNT = my_vector_alloc(2*NFFT);
	  pulsars[i].FNF = my_matrix_alloc(2*NFFT,2*NFFT);

#else
	  pulsars[i].phi_inv = my_matrix_alloc(NFFT,NFFT);
	  pulsars[i].F = my_matrix_alloc(pulsars[i].N,NFFT);
	  pulsars[i].GF = my_matrix_alloc(pulsars[i].N_m,NFFT);
	  pulsars[i].FNT = my_vector_alloc(NFFT);
	  pulsars[i].FNF = my_matrix_alloc(NFFT,NFFT);
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
      //      printf("Computing Gres %d again\n",i);
      my_dgemv(CblasTrans,1.0,pulsars[i].G,pulsars[i].res,0.0,pulsars[i].Gres);
      //      my_vector_print(pulsars[i].Gres);
      if (only_res == 0)
	{
#ifndef POWERLAW
	  *alldim = totstride;
#else
	  *alldim = 0;
#ifdef GWB
	  *alldim += 2;
#endif
#ifdef REDNOISE
	  *alldim += 2;
#endif
#ifdef DM
	  *alldim += 2;
#endif
#endif
	}

    }

  if (only_res == 0)
    {
      double tmin= 1.0E10;
      double tmax = 0.0;
      //  double temp;
      int a;
      int b;
      for (a = 0; a < Nplsr; a++)
	  for (b = 0; b < pulsars[a].N; b++)
	    {
	      if ((pulsars[a].toa->data[b]) > tmax)
		tmax = pulsars[a].toa->data[b];
	      if ((pulsars[a].toa->data[b]) < tmin)
		tmin = pulsars[a].toa->data[b];
	    }
      temp = 2.0*(tmax - tmin);
      //temp = *tspan*1.306;                                                                                                                                                 
      //  temp = *tspan;
      //temp = *tspan*1.18;                                                                                                                                                  
      double ffund = 1.0/(1.0*temp);
      //set up frequency grid                                                                                                                                                
      printf("FREQS -1\t%e\n",b,temp);
      for (a = 0; a < Nplsr; a++)
	{
	  for (b = 0; b < (NFFT/2); b++)
	    {
	      pulsars[a].freqs[b] = ffund*(b+1);
	      if (a == 0)
		printf("FREQS %d\t%e\n",b,pulsars[a].freqs[b]);
	    }
	  pulsars[a].tspan = temp;
	}
    }


  if (verbose)
    printf("Ndim is %d\n",*Ndim);
}

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
	  //	  printf("FREQ %s\t%d  %g\n",psr->name,i,f);
	  int row = i*2;
	  int row_cos = i*2+1;
	  psr->F->data[(row)*psr->F->m + b] = sin(2.0*PI*psr->toa->data[b]*f);///par->tspan;
	  psr->F->data[(row_cos)*psr->F->m + b] = cos(2.0*PI*psr->toa->data[b]*f);///par->tspan;
	  //	  printf("%g\n",psr->F->data[(row)*psr->F->m + b]*psr->F->data[(row)*psr->F->m + b]+psr->F->data[(row_cos)*psr->F->m + b]*psr->F->data[(row_cos)*psr->F->m + b]);
#ifdef DM
	  psr->F->data[(row+NFFT)*psr->F->m + b] = 1.0/(K_DM*psr->obsfreqs->data[b]*psr->obsfreqs->data[b])*sin(2.0*PI*psr->toa->data[b]*f);///par->tspan;
	  psr->F->data[(row_cos+NFFT)*psr->F->m +  b] = 1.0/(K_DM*psr->obsfreqs->data[b]*psr->obsfreqs->data[b])*cos(2.0*PI*psr->toa->data[b]*f);///par->tspan;
#endif	  
	}
    }
  my_dgemm(CblasTrans,CblasNoTrans,1.0,psr->G,psr->F,0.0,psr->GF);
}

//Get FFT matrix for one single pulsar given some DM for this backend
//void initialize_fft(struct my_matrix * F, struct mypulsar * psr, int Nplsr, int Ntot, struct parameters * par)
//{
//  int i,a,b;
//  double f;
//
//  int ind_tot = 0;
//  int ind_r = 0;
//
//  for (a = 0; a < Nplsr; a++)
//    {
//    for (b = 0; b < psr[a].N; b++)
//      {
//	for (i = 0; i < (NFFT/2); i++)
//	  {
//	    f = psr[a].freqs[i];
//	    int row = i*2;
//	    int row_cos = i*2+1;
//	    F->data[(ind_r+row)*F->m + ind_tot + b] = sin(2.0*PI*psr[a].toa->data[b]*f);///par->tspan;
//	    F->data[(ind_r+row_cos)*F->m + ind_tot + b] = cos(2.0*PI*psr[a].toa->data[b]*f);///par->tspan;
//#ifdef DM
//	    F->data[(ind_r+row+NFFT)*F->m + ind_tot + b] = 1.0/(K_DM*psr[a].obsfreqs->data[b]*psr[a].obsfreqs->data[b])*sin(2.0*PI*psr[a].toa->data[b]*f);///par->tspan;
//	    F->data[(ind_r+row_cos+NFFT)*F->m +  ind_tot +b] = 1.0/(K_DM*psr[a].obsfreqs->data[b]*psr[a].obsfreqs->data[b])*cos(2.0*PI*psr[a].toa->data[b]*f);///par->tspan;
//#endif	  
//	  }
//	ind_tot++;
//      }
//    ind_r += NFFT;
//    }
//  //  my_matrix_print("F\0",psr->F);
//  //  my_dgemm(CblasTrans,CblasNoTrans,1.0,psr->G,psr->F,0.0,psr->GF);
//}

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
  double rfac = rAgw * rAgw / (12.0*PI*PI) * 3.16e22;
#ifdef DM
  double dmAgw = psr->dmA;
  double dmgamma = psr->dmgamma;
  double dmfac = dmAgw * dmAgw / (12.0*PI*PI) * 3.16e22;
#endif
  for (i = 0; i < NFFT/2; i++)
    {
      f = psr->freqs[i];
      //      power = rfac* pow(f/3.17e-08,-rgamma) / par.tspan;
      //add red noise for diagonal elements
      power = rfac * pow(f/3.17e-08,-rgamma)/psr->tspan;
      psr->phi_inv->data[(2*i)*psr->phi_inv->m  + 2*i] = 1.0/power;
      psr->phi_inv->data[(2*i+1)*psr->phi_inv->m + 2*i + 1] = 1.0/power;
#ifdef DM
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
  if (verbose == 2)
    my_matrix_print("FFT\0",psr->F);
  tend = omp_get_wtime();
  if (verbose == 3)
    printf("compute_fft\t\t\t%g\n",tend-tstart);
  //biuld phi_inv directly, only diagonal
//  if (verbose ==2)
//    printf("Compute phi\n");
//  tstart = omp_get_wtime();
//  calculate_phi_inv_per_pulsar(psr,*par);
//  tend = omp_get_wtime();
//  if (verbose == 3)
//    printf("compute_phi\t\t\t%g\n",tend-tstart);
  //	  my_matrix_print("phi inv\0",psr->phi_inv);
  //build up F N F
  if (verbose ==2)
    printf("Compute FNF\n");
  struct my_matrix *FN;
#ifdef DM
  FN = my_matrix_alloc(2*NFFT,psr->N_m);
#else
  FN = my_matrix_alloc(NFFT,psr->N_m);
#endif
  my_dgemm(CblasTrans,CblasNoTrans,1.0,psr->GF,psr->GNGinv,0.0,FN);
  my_dgemm(CblasNoTrans,CblasNoTrans,1.0,FN,psr->GF,0.0,psr->FNF);
  my_dgemv(CblasNoTrans,1.0,FN,psr->Gres,0.0,psr->FNT);
  // end FNF
  //compute tNt

  struct my_vector * temp;
  temp = my_vector_alloc(psr->N_m);
  my_dgemv(CblasNoTrans,1.0,psr->GNGinv,psr->Gres,0.0,temp);
  //  my_vector_print(psr->Gres);
  my_vector_mult(temp,psr->Gres,&(psr->tNt));
  my_vector_free(temp);
  my_matrix_free(FN);
  //add to phi_inv and invert
//  
//  if (verbose ==2)
//    printf("invert FNF plus phi\n");
//  struct my_matrix * FNF_lu;
//#ifdef DM
//  FNF_lu = my_matrix_alloc(2*NFFT,2*NFFT);
//#else
//  FNF_lu = my_matrix_alloc(NFFT,NFFT);
//#endif
//
//  my_matrix_add(FNF,psr->phi_inv);
//  
//  double detFNF;
//#ifdef DM
//  get_inverse_lu(FNF,FNF_lu,2*NFFT,&detFNF);
//#else
//  get_inverse_lu(FNF,FNF_lu,NFFT,&detFNF);
//#endif
//  
//  //reuse F*N
//  struct my_matrix *FNFFN;
//#ifdef DM
//  FNFFN = my_matrix_alloc(2*NFFT,psr->N_m);
//#else
//  FNFFN = my_matrix_alloc(NFFT,psr->N_m);
//#endif
//  my_dgemm(CblasNoTrans,CblasNoTrans,1.0,FNF,FN,0.0,FNFFN);
//  my_dgemm(CblasTrans,CblasNoTrans,1.0,FN,FNFFN,0.0,psr->Cinv);
//  //my_matrix_print("Nwiggle\0",pulsars[i].Nwiggle);
//  //      my_matrix_print("C\0",C);
//  
//#ifdef REDNOISE
//  my_matrix_sub(psr->GNGinv,psr->Cinv);
//#else
//  my_matrix_memcpy(psr->Cinv,psr->GNGinv);
//#endif
//      //now C is C^-1!
//  my_matrix_free(FNFFN);
//  my_matrix_free(FNF_lu);
//  my_matrix_free(FN);
//  my_matrix_free(FNF);

}

struct geo_par geo_fac(struct parameters * par, struct mypulsar * pulsars,int a, int b)
{

  int offset = 2;
#ifndef POWERLAW
  offset = NFFT/2;
#ifdef REDNOISE
  offset = NFFT;
#endif
#endif
  struct geo_par fac;

  double thetaS = par->values[offset+0];
  double phiS = par->values[offset+1];

#ifdef ANGLES
  double psi = par->values[offset+3];
  double iota = par->values[offset+4];
  double phi0 = par->values[offset+5];
#else
  double psi = 2.413;
  double iota = 0.648;
  double phi0 = 1.565;
#endif

  double theta_a = 0.5*PI - pulsars[a].dec;
  double phi_a = pulsars[a].raj;
  double theta_b = 0.5*PI - pulsars[b].dec;
  double phi_b = pulsars[b].raj;


  double cthS = cos(thetaS);
  double sthS = sin(thetaS);
  double cphS = cos(phiS);
  double sphS = sin(phiS);

  //Compute k,u,v vector
  double k[3],u[3],v[3];
  k[0] = -cphS*sthS;
  k[1] = -sphS*sthS;
  k[2] = -cthS;
  
  u[0] = cphS*cthS;
  u[1] = sphS*cthS;
  u[2] = -sthS;

  v[0] = sphS;
  v[1] = -cphS;
  v[2] = 0.0;

  //Pulsar a angles                                                                                                                                                      
  double n_a[3];
  n_a[0] = cos(phi_a)*sin(theta_a);
  n_a[1] = sin(phi_a)*sin(theta_a);
  n_a[2] = cos(theta_a);
  // k * n                                                                                                                                                               
  double kn_a = k[0]*n_a[0] + k[1]*n_a[1] + k[2]*n_a[2];
  // u * n                                                                                                                                                               
  double un_a = u[0]*n_a[0] + u[1]*n_a[1] + u[2]*n_a[2];
  // u * n                                                                                                                                                               
  double vn_a = v[0]*n_a[0] + v[1]*n_a[1] + v[2]*n_a[2];
  double beta_a = acos(-kn_a);
  //  printf("A %g\n",vn_a/un_a);                                                                                                                                        
  double alpha_a = atan(vn_a/un_a);

  //Pulsar b angles                                                                                                                                                      
  double n_b[3];
  n_b[0] = cos(phi_b)*sin(theta_b);
  n_b[1] = sin(phi_b)*sin(theta_b);
  n_b[2] = cos(theta_b);
  // k * n                                                                                                                                                               
  double kn_b = k[0]*n_b[0] + k[1]*n_b[1] + k[2]*n_b[2];
  // u * n                                                                                                                                                               
  double un_b = u[0]*n_b[0] + u[1]*n_b[1] + u[2]*n_b[2];
  // u * n                                                                                                                                                               
  double vn_b = v[0]*n_b[0] + v[1]*n_b[1] + v[2]*n_b[2];
  double beta_b = acos(-kn_b);
  //printf("B %g\n",vn_b/un_b);                                                                                                                                          
  double alpha_b = atan(vn_b/un_b);

  double F_cross_a = (1.0+cos(beta_a))*cos(2.0*alpha_a);
  double F_cross_b = (1.0+cos(beta_b))*cos(2.0*alpha_b);
  double F_plus_a = (1.0+cos(beta_a))*sin(2.0*alpha_a);
  double F_plus_b = (1.0+cos(beta_b))*sin(2.0*alpha_b);

  double y_a = (F_plus_a*sin(2.0*psi) + F_cross_a*cos(2.0*psi));
  double y_b = (F_plus_b*sin(2.0*psi) + F_cross_b*cos(2.0*psi));

  double x_a = (F_plus_a*cos(2.0*psi) - F_cross_a*sin(2.0*psi));
  double x_b = (F_plus_b*cos(2.0*psi) - F_cross_b*sin(2.0*psi));

  double amp = pow(10.0,par->values[offset+2]);

  double a_plus2 = (amp*(1.0+cos(iota)*cos(iota)))*(amp*(1.0+cos(iota)*cos(iota)));
  double a_cross2 = (2.0*amp*cos(iota))*(2.0*amp*cos(iota));

  double cos2phi = cos(phi0)*cos(phi0);
  double sin2phi = sin(phi0)*sin(phi0);

  fac.Fac = a_plus2*cos2phi*x_a*x_b + a_cross2*sin2phi*y_a*y_b;
  fac.Fas = a_plus2*sin2phi*x_a*x_b + a_cross2*cos2phi*y_a*y_b;


  return fac;
}

void calculate_phi(struct my_matrix * a_ab, struct my_matrix * phi,struct parameters params, int Nplsr, struct mypulsar * pulsars)
{
  int i,a,b;
  double f;
  int ind_r,ind_c;
  double power;
  int offset = 0;
#ifdef GWB
  offset = 2;
#endif
  my_matrix_set_zero(phi);
#ifdef POWERLAW
#ifdef GWB
  double Agw = params.values[0];
  double fac = Agw * Agw / (12.0*PI*PI) * 3.16e22;
#endif
#endif

#ifdef SS
  struct geo_par g_fac;
#endif
  for (a = 0; a < Nplsr; a++)
    {
      for (b = 0; b < Nplsr; b++)
        {
          ind_r = NFFT*a;
          ind_c = NFFT*b;
          for (i = 0; i < NFFT/2; i++)
            {
              f = pulsars[a].freqs[i]; // or b because they have to be the same
#ifdef GWB
#ifdef POWERLAW
              //f = (float) (i+1) * ffund;                                                                                                                               
              power =  a_ab->data[b*a_ab->m + a]*fac* pow(f/3.17e-08,-params.values[1])/params.tspan;
#else
              power =  a_ab->data[b*a_ab->m + a]*pow(10.0,params.values[i]);
#endif
              //printf("PHI\t%g\t%g\n",f,fac* pow(f/3.17e-08,-4.333) /params.tspan);                                                                                            
              phi->data[(ind_c+2*i)*phi->m + ind_r + 2*i] = power;
              phi->data[(ind_c+2*i+1)*phi->m + ind_r + 2*i + 1] = power;
#endif
#ifdef SS
              if (i == (NFFT/2-1))
                {
                  g_fac = geo_fac(&params,pulsars,a,b);
//		  if ((a == 0) && (b == 0))
//		    printf("adding geofac %g to %g\n",g_fac.Fas,phi->data[(ind_c+2*i)*phi->m + ind_r + 2*i]);
                  phi->data[(ind_c+2*i)*phi->m + ind_r + 2*i] +=  g_fac.Fac;
                  phi->data[(ind_c+2*i+1)*phi->m + ind_r + 2*i + 1] += g_fac.Fas;
                }
#endif

#ifdef REDNOISE
              if (a == b)
                {
#ifdef POWERLAW
		  double rAgw = pulsars[a].rA;
		  double rgamma = pulsars[a].rgamma;
		  double rfac = rAgw * rAgw / (12.0*PI*PI) * 3.16e22;
		  if (verbose)
		    printf("Adding red noise %g\t%g\n",rfac,rgamma);
		  power =  rfac* pow(f/3.17e-08,rgamma)/params.tspan;
#else
		  power = pow(10.0,pulsars[a].rnamp[i]);
#endif
		  phi->data[(ind_c+2*i)*phi->m + ind_r + 2*i] += power;
		  phi->data[(ind_c+2*i+1)*phi->m + ind_r + 2*i + 1] += power;

#ifdef DM
#ifdef POWERLAW
		  double dmAgw = pulsars[a].dmA;
		  double dmgamma = pulsars[a].dmgamma;
		  double dmfac = dmAgw * dmAgw / (12.0*PI*PI) * 3.16e22;
		  power = dmfac * pow(f/3.17e-08,-dmgamma)/params.tspan;
#else
		  power = pow(10.0,pulsars[a].dmamp[i]);
#endif
		  phi->data[(NFFT*Nplsr+ind_c+2*i)*phi->m  + NFFT*Nplsr + ind_r + 2*i] = power;
		  phi->data[(NFFT*Nplsr+ind_c+2*i+1)*phi->m + NFFT*Nplsr + ind_r + 2*i + 1] = power;
#endif

//
//                  phi->data[(ind_c+2*i)*phi->m + ind_r + 2*i] += power;
//                  phi->data[(ind_c+2*i+1)*phi->m + ind_r + 2*i + 1] += power;
                }
#endif
            }
        }
    }
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
      if ((2.0*1.0/pulsars[i].tspan) > (par->omega/(2.0*PI)))
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

void ObjFuncRandomizePoint(ObjFunc *S, double *x)
{
  int i;
  for (i = 0; i < S->dim; i++)
    {
      x[i] = gsl_ran_flat(S->r, S->l[i], S->u[i]);
      if (verbose)
	printf("Randomized %d\t%f\t%f\t%f\n",i,S->l[i],x[i],S->u[i]);
    }
}

void ObjFuncInitLimits(ObjFunc *S)
{
  int i,j;
  int Nplsr = S->Nplsr;
#ifndef POWERLAW
#ifdef GWB
  for (i = 0; i < NFFT/2; i++)
    {
      S->l[i] = -20.0; // GWB
      S->u[i] = -8.0; // GWB
    }
#endif
  for (i = 0; i < S->Nplsr; i++)
    {
      int stride = S->pulsars[i].stride;
      if (verbose)
	printf("Strideeeeeeeeeeeeeeeeeeeeeeeeee %d\t%d\n",i,stride);
      for (j = 0; j < NFFT/2; j++)
	{
	  S->l[stride + j] = -20.0;
	  S->l[stride + NFFT/2 + j] = -20.0;
	  S->u[stride + j] = -8.0;
	  S->u[stride + NFFT/2 + j] = -8.0;
	}
      for (j = 0; j < S->pulsars[i].n_be; j++)
	{
	  S->l[stride + NFFT + j] = 0.1;
	  S->u[stride + NFFT + j] = 5.0;
	  S->l[stride + NFFT + S->pulsars[i].n_be + j] = -12.0;
	  S->u[stride + NFFT + S->pulsars[i].n_be + j] = -5.0;
	}
    }
#else
  int offset = 0;
#ifdef GWB
  S->l[offset] = -16.0;
  S->u[offset] = -12.0;
  S->l[offset+1] = 3.0;
  S->u[offset+1] = 6.0;
  offset += 2;
#endif
#ifdef REDNOISE
  S->l[offset] = -20.0;
  S->u[offset] = -3.0;
  S->l[offset+1] = 0.0;
  S->u[offset+1] = 5.0;
  offset += 2;
#endif
#ifdef DM
  S->l[offset] = -20.0;
  S->u[offset] = -5.0;
  S->l[offset+1] = 0.0;
  S->u[offset+1] = 5.0;
#endif
#endif

}

double ObjFuncEval(ObjFunc *S, double *x, int prime)
{
  //double compute_likelihood(struct my_matrix * phi,struct mypulsar * pulsars, int Nplsr)
  //{
  //copy all params x into appropriate places
  int i,j;
  int Nplsr = S->Nplsr;
#ifndef POWERLAW
#ifdef GWB
  for (i = 0; i < NFFT/2; i++)
    S->params.values[i] = x[i]; // GWB
#endif
  for (i = 0; i < S->Nplsr; i++)
    {
      int stride = S->pulsars[i].stride;
      for (j = 0; j < NFFT/2; j++)
	{
	  S->pulsars[i].rnamp[j] = x[stride + j];
	  S->pulsars[i].dmamp[j] = x[stride + NFFT/2 + j];
	}
      for (j = 0; j < S->pulsars[i].n_be; j++)
	{
	  S->pulsars[i].efac[j] = x[stride + NFFT + j];
	  S->pulsars[i].equad[j] = x[stride + NFFT + S->pulsars[i].n_be + j];
	}
    }
#else
  int offset = 0;
#ifdef GWB
  S->params.values[0] = pow(10.0,x[0]);
  S->params.values[1] = x[1];
  offset = 2;
#endif
#ifdef REDNOISE
  for (i = 0; i < S->Nplsr; i++)
    {
      S->pulsars[i].rA = pow(10.0,x[offset]);
      S->pulsars[i].rgamma = x[offset+1];
    }
  offset +=2;
#endif
#ifdef DM
  for (i = 0; i < S->Nplsr; i++)
    {
      S->pulsars[i].dmA = pow(10.0,x[offset]);
      S->pulsars[i].dmgamma = x[offset+1];
    }
#endif
#endif
  //rebuild FNF and stuff
  for (i = 0; i < Nplsr; i++)
    {
      compute_C_matrix(&(S->pulsars[i]),&(S->params));
    }

  calculate_phi(S->a_ab,S->phi,S->params,S->Nplsr,S->pulsars);

  double likeli = 0.0;
  double tNt = 0.0;
  //get inverse of phi                                                                                                                                                   
  struct my_matrix * phiinv;
  struct my_matrix * sigmainv;
  struct my_matrix * cholesky_phi;
  struct my_matrix * cholesky_sigma;
#ifdef DM
  phiinv = my_matrix_alloc(2*NFFT*Nplsr,2*NFFT*Nplsr);
  sigmainv = my_matrix_alloc(2*NFFT*Nplsr,2*NFFT*Nplsr);
  cholesky_phi = my_matrix_alloc(2*NFFT*Nplsr,2*NFFT*Nplsr);
  cholesky_sigma = my_matrix_alloc(2*NFFT*Nplsr,2*NFFT*Nplsr);
#else
  phiinv = my_matrix_alloc(NFFT*Nplsr,NFFT*Nplsr);
  sigmainv = my_matrix_alloc(NFFT*Nplsr,NFFT*Nplsr);
  cholesky_phi = my_matrix_alloc(NFFT*Nplsr,NFFT*Nplsr);
  cholesky_sigma = my_matrix_alloc(NFFT*Nplsr,NFFT*Nplsr);
#endif
  my_matrix_memcpy(phiinv,S->phi);
  //my_matrix_memcpy(sigmainv,FNF);
  
  double  detPhi = 0.0;
  double detSigma = 0.0;

  //  my_matrix_print("phi\0",phi);

//  if (get_inverse_cholesky(phiinv,cholesky_phi,NFFT*Nplsr) == 0 && (verbose))                             
//    printf("My inversion of phi work\n");
#ifdef DM
  int dmtemp = 2;

  if (verbose)
    {
      printf("Inverting phi\n");
      my_matrix_print("Phi\0",phiinv);
    }
  if (get_inverse_lu(phiinv,cholesky_phi,2*NFFT*Nplsr,&detPhi) == 0 && (verbose))
    printf("My inversion of phi work\n");
#else
  int dmtemp = 1;
  if (get_inverse_lu(phiinv,cholesky_phi,NFFT*Nplsr,&detPhi) == 0 && (verbose))
    printf("My inversion of phi work\n");
#endif
  //construct phiinv + FNF object
  int a,b;
  int ind_c = 0,ind_r = 0;
  my_matrix_memcpy(sigmainv,phiinv);
  for (a = 0; a < Nplsr; a++)
    {
      for (i = 0; i < (dmtemp*NFFT); i++)
	for (j = 0; j < (dmtemp*NFFT); j++)
	  sigmainv->data[(i + ind_c)*sigmainv->m + ind_c + j] += S->pulsars[a].FNF->data[i*S->pulsars[a].FNF->m + j];
      ind_c += (dmtemp*NFFT);
    }

  //invert Sigma
#ifdef DM
  if (get_inverse_lu(sigmainv,cholesky_sigma,2*NFFT*Nplsr,&detSigma) == 0 && (verbose))
    printf("My inversion of sigma worked\n");
#else
  if (get_inverse_lu(sigmainv,cholesky_sigma,NFFT*Nplsr,&detSigma) == 0 && (verbose))
    printf("My inversion of sigma worked\n");
#endif
  struct my_vector * dsigma;
  struct my_vector * FNTbig;
#ifdef DM
  dsigma = my_vector_alloc(2*NFFT*Nplsr);
  FNTbig = my_vector_alloc(2*NFFT*Nplsr);
#else
  dsigma = my_vector_alloc(NFFT*Nplsr);
  FNTbig = my_vector_alloc(NFFT*Nplsr);
#endif
  for (a = 0; a < Nplsr; a++)
    for (i = 0; i < (dmtemp*NFFT); i++)
      FNTbig->data[a*(dmtemp*NFFT) + i] = S->pulsars[a].FNT->data[i];
  double dsd;
  //combine FNT into FNTbig vector
  my_dgemv(CblasNoTrans,1.0,sigmainv,FNTbig,0.0,dsigma); 
  my_vector_mult(FNTbig,dsigma,&dsd);
  //add together tNt per pulsar
  double detN = 0.0;
  for (i = 0; i < Nplsr; i++)
    {
      tNt += S->pulsars[i].tNt;
      detN += S->pulsars[i].det;
    }

  likeli = tNt-dsd;
  if (verbose)
    printf("tnt dsd\t%g\t%g\n",tNt,dsd);
  if (verbose)
    printf("detPhi is %g \n",detPhi);
  likeli += detPhi;
  //  for (i = 0; i < (NFFT*Nplsr); i++)                                                                                                                                 
  //  detSigma += 2.0*log(cholesky_sigma->data[i*NFFT*Nplsr+i]);                                                                                                         
  likeli += detSigma;
  if (verbose)
    printf("detSigma is %g \n",detSigma);
  //add det N                                                                                                                                                            
  likeli += detN ;
  //printf("likelihood is %g after detn \n",likeli);                                                                                                                     

  my_matrix_free(sigmainv);
  my_matrix_free(phiinv);
  my_matrix_free(cholesky_phi);
  my_matrix_free(cholesky_sigma);
  my_vector_free(dsigma);
  my_vector_free(FNTbig);

  return (likeli);

}

//void initialize_FNF(struct my_matrix * FNF, struct my_vector * FNT, struct my_matrix * Nwiggle_large, double *tNt, struct mypulsar * pulsars, struct my_matrix * F, int Ntot, int Nplsr, double * det, int Ndim)
//{
//  int a,b,c,ind_c,ind_r;
//  struct my_matrix *temp;
//  temp = my_matrix_alloc(NFFT*Nplsr,Ntot);
//  //make a large N which will have many zeros but anyway                                                                                                                 
//  ind_r = 0; ind_c = 0;
//  for (a = 0; a < Nplsr; a++)
//    {
//      for (b = 0; b < pulsars[a].N; b++)
//        {
//          for (c = 0; c < pulsars[a].N; c++)
//            {
//              Nwiggle_large->data[(ind_c+c)*Nwiggle_large->m + ind_r + b] = pulsars[a].Nwiggle->data[c*pulsars[a].Nwiggle->m+b];
//              //              gsl_matrix_set(Nwiggle_large,ind_r+b,ind_c+c,gsl_matrix_get(pulsars[a].Nwiggle,b,c));                                                      
//              //              printf("setting a %g\n",gsl_matrix_get(pulsars[a].Nwiggle,b,c));                                                                           
//            }
//        }
//      ind_r += pulsars[a].N;
//      ind_c += pulsars[a].N;
//    }
//
//
//  ind_r = 0; ind_c = 0;
//  if (verbose)
//    printf("Multiplying F * Nwiggle\n");
//  my_dgemm(CblasTrans,CblasNoTrans,1.0,F,Nwiggle_large,0.0,temp);
//  if (verbose)
//    printf("Multiplying F * Nwiggle * F\n");
//  my_dgemm(CblasNoTrans,CblasNoTrans,1.0,temp,F,0.0,FNF);
//  //make one large vector for all the residuals                                                                                                                          
//  struct my_vector * TOAlarge;
//  TOAlarge = my_vector_alloc(Ntot);
//  ind_c = 0;
//  for (a = 0; a < Nplsr; a++)
//    {
//      for (b = 0; b < pulsars[a].N; b++)
//        {
//          TOAlarge->data[ind_c+b] = pulsars[a].res[b];
//        }
//      ind_c += pulsars[a].N;
//    }
//  if (verbose)
//    printf("Multiplying FNT\n");
//  my_dgemv(CblasNoTrans,1.0,temp,TOAlarge,0.0,FNT);
//  struct my_vector * temp2;
//  temp2 = my_vector_alloc(Ntot);
//  if (verbose)
//    printf("Multiplying Nt\n");
//  my_dgemv(CblasNoTrans,1.0,Nwiggle_large,TOAlarge,0.0,temp2);
//  my_vector_mult(TOAlarge,temp2,tNt);
//  //invert Nwiggle                                                                                                                                                       
//  struct my_matrix * GNG_lu;
//  GNG_lu = my_matrix_alloc(Ntot,Ntot);
//  get_inverse_lu(Nwiggle_large,GNG_lu,Ntot,det);
//  my_matrix_free(GNG_lu);
////  *det = 0;                                                                                                                                                            
//////  for (i = 0; i < Ndim; i++)                                                                                                                                         
//////    {                                                                                                                                                                
////////      *det += log(gsl_matrix_get(GNG_large,i,i));                                                                                                                  
//////  printf("Determinant is\t%g\n",gsl_matrix_get(GNG_large,i,i));                                                                                                      
//////  //  printf("Determinant is %g\n",*det);                                                                                                                            
//////      }                                                                                                                                                              
//////    printf("%d\t%g\n",i,gsl_matrix_get(Nwiggle_large,i,i));                                                                                                          
//////  *det = gsl_linalg_LU_det(Nwiggle_large,*signum);                                                                                                                   
//  //gsl_matrix_free(Cholesky);                                                                                                                                           
//  my_matrix_free(temp);
//  my_vector_free(TOAlarge);
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
#ifdef GRID
void maximize_likelihood(struct my_matrix *a_ab, struct my_matrix *phi,double detGNG,int Nplsr,struct parameters * params,struct mypulsar * pulsars)
{
  double likeli_old;

  params->values[0] = -13.3;
  params->values[1] = 4.333;
  int offset = 2;
#ifdef SS
  params->values[offset+0] = 1.0;
  params->values[offset+1] = 1.5;
  params->values[offset+2] = -8.5;
  params->values[offset+3] = 1.0;
  params->values[offset+4] = 0.8;
  params->values[offset+5] = 2.5;
#endif
  calculate_phi(a_ab,phi,*params,Nplsr,pulsars);
  likeli_old = compute_likelihood(phi,pulsars,Nplsr);
  //  printf("#VALUE AT %f\t%f\t%f\n",params->values[6],params->values[7],likeli_old);
  double minimum = 1e10;
  //for (params->values[offset+2] = params->l[offset+2]; params->values[offset+2] < params->u[offset+2]; params->values[offset+2] += (params->u[offset+2]-params->l[offset+2])/150.0)
  offset = 0;
  //printf("%f\t%f\t%f\n",params->values[offset+0],params->l[offset+0],params->u[offset+0]);
  //  for (params->values[offset+2] = params->l[offset+2]; params->values[offset+2] < params->u[offset+2]; params->values[offset+2] += (params->u[offset+2]-params->l[offset+2])/50.0)
    for (params->values[offset+0] = params->l[offset+0]; params->values[offset+0] < params->u[offset+0]; params->values[offset+0] += (params->u[offset+0]-params->l[offset+0])/30.0)
      for (params->values[offset+1] = params->l[offset+1]; params->values[offset+1] < params->u[offset+1]; params->values[offset+1] += (params->u[offset+1]-params->l[offset+1])/30.0)
     //for (params->values[offset-2+6] = params->l[offset-2+6]; params->values[offset-2+6] < params->u[offset-2+6]; params->values[offset-2+6] += (params->u[offset-2+6]-params->l[offset-2+6])/20.0)                                                                                                                                            
     //for (params->values[offset-2+7] = params->l[offset-2+7]; params->values[offset-2+7] < params->u[offset-2+7]; params->values[offset-2+7] += (params->u[offset-2+7]-params->l[offset-2+7])/20.0)                                                                                                                                            
	{
        calculate_phi(a_ab,phi,*params,Nplsr,pulsars);
        //      my_matrix_print("PHI\0",phi);                                                                                                                            
        likeli_old = compute_likelihood(phi,pulsars,Nplsr);
        if (likeli_old < minimum)
          {
            minimum = likeli_old;
            printf("#NEW MIN\n");
          }
        //printf("%f\t%f\t%f\t%f\n",params->values[offset+2],params->values[offset+0],params->values[offset+1],likeli_old);                                        
	printf("%f\t%f\t%f\n",params->values[offset+0],params->values[offset+1],likeli_old);
        //printf("%f\t%f\n",params->values[offset+2],likeli_old);                                                                                                        
        fflush(stdout);
      }


}
#endif

#ifdef MPI
struct particle * init_parts(int n, struct parameters params)
{
  int i,j,k;
  struct particle * allparts;
  allparts = (struct particle *) malloc(n*sizeof(struct particle));
  double vmax[NCOEFF];
  for (j = 0; j < NCOEFF; j++)
    vmax[j] = 0.5* (params.u[j] - params.l[j]);
  //  vmax[1] = 0.5*(params.bound_gamma_gw[1] - params.bound_gamma_gw[0]);                                                                                               
  for (i = 0; i < n; i++)
    {
      for (j = 0; j < NCOEFF; j++)
        {
          allparts[i].x[j] = (double) rand()/RAND_MAX * (params.u[j] - params.l[j]) + params.l[j];
          allparts[i].v[j] = (double) rand()/RAND_MAX * 2.0 * vmax[j] - vmax[j];
          allparts[i].pbest[j] = allparts[i].x[j];
          allparts[i].mean[j] = allparts[i].x[j];
          //set memory to 0                                                                                                                                              
          allparts[i].memory[j][0] = allparts[i].x[j];
	  for (k = 1; k < NMEM; k++)
            allparts[i].memory[j][k] = 0.0;
        }
    }
  return allparts;
}


void maximize_likelihood_mpi(struct my_matrix *a_ab, struct my_matrix *phi,int Nplsr,struct parameters * params, struct mypulsar *pulsars)
{
  struct particle l_part[NPERNODE];

  //initialize NSWARM particles with random positions and velocities
  int rank,nproc;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  struct particle * parts;

  int offset = 2;
#ifdef REDNOISE
  offset = 4;
#endif
#ifndef POWERLAW
  offset = NFFT/2;
#ifdef REDNOISE
  offset = NFFT;
#endif
#endif

  int i,j,k;

  if (rank == 0)
    {
      parts = init_parts(nproc*NPERNODE, *params);
      //what the master has in its array
      //      for (i = 0; i < nproc; i++)
      //{
      //  printf("Master: %d\t%g\t%g\n",i,parts[i].x[0],parts[i].v[1]);
      //}
    }
  MPI_Scatter(parts,NPERNODE*sizeof(struct particle),MPI_BYTE,l_part,NPERNODE*sizeof(struct particle), MPI_BYTE,0, MPI_COMM_WORLD);
  //  printf("Nodes: %d\t%g\t%g\n",rank,l_part.x[0],l_part.v[1]);

  //initialize pmax and gmax likelihood values;
  double gmax = 10000000000.0;
  double vmax[NCOEFF];
  for (j = 0; j < NCOEFF; j++)
    vmax[j] = 0.5*(params->u[j] - params->l[j]);
  //calc pbest for each node

  double gbest[NCOEFF];
  //  double psi[NCOEFF];
  //  for (i = 0; i < nproc; i++)
  //  {
  int npn;
  for (npn = 0; npn < NPERNODE; npn++)
    {
      for (j = 0; j < NCOEFF; j++)
	params->values[j] = l_part[npn].x[j];
      calculate_phi(a_ab,phi,*params,Nplsr,pulsars);
      l_part[npn].pmax = compute_likelihood(phi,pulsars,Nplsr);
    }
  //now copy parts to root and find gbest at root
  MPI_Gather(l_part,NPERNODE*sizeof(struct particle),MPI_BYTE,parts,NPERNODE*sizeof(struct particle), MPI_BYTE, 0, MPI_COMM_WORLD);
  if (rank == 0)
    for (i = 0; i < (NPERNODE*nproc); i++)
      if (parts[i].pmax < gmax)
	{
	  gmax = parts[i].pmax;
	  for (j = 0; j < NCOEFF; j++)
	    gbest[j] = parts[i].pbest[j];
	}
  //Bcast gbest. Nodes dont need to know the gmax
  MPI_Bcast(gbest,NCOEFF,MPI_DOUBLE,0,MPI_COMM_WORLD);
  //enter loop of iterations
  double c1 = 1.193; double c2 = 1.193; double w = 0.72;
  double likelihood;
  double u1,u2;
  int iter;
  double psi[NCOEFF],bovern[NCOEFF],W[NCOEFF];
  double sigma_plus[NCOEFF],Rhat[NCOEFF];

  int switched = 0;//indicator if i just placed a new gbest from a switch in rank 0 particle
  //  params->ifreq = params->ifreq;
  //perform a walk for each particle
  double ss_bayes = -100.0;
  for (iter = 0; iter < NITER; iter++)
    {
      if (rank == 0)
	{
	  printf("GBEST ");
	  for (j = 0; j < NCOEFF; j++)
	    printf("% 3.3f  ",gbest[j]);//,gbest[1],gmax);
	  //compute what the likelihood would be without the source
	  for (j = 0; j < NCOEFF; j++)
	    {
	      params->values[j] = gbest[j]; 
	    }
	  //set amplitude to very low
	  params->values[offset+2] = -20.0;
	  calculate_phi(a_ab,phi,*params,Nplsr,pulsars);
	  
	  likelihood = compute_likelihood(phi,pulsars,Nplsr);
	  printf("% 3.3f  ",likelihood-gmax);  
	  printf("% 3.3f\n",gmax);
	  ss_bayes = likelihood-gmax;
	}
      for (npn = 0; npn < NPERNODE; npn++)
	{
	  u1 = (double) rand()/RAND_MAX;
	  u2 = (double) rand()/RAND_MAX;
      //      if (verbose)
      //{
      //  printf("u1: %g\tu2:%g\n",u1,u2);
      //}
	  for (j = 0; j < NCOEFF; j++)
	    {
	      l_part[npn].v[j] = w * l_part[npn].v[j] + c1 * u1 * (l_part[npn].pbest[j] - l_part[npn].x[j]) + c2 * u2 * (gbest[j] - l_part[npn].x[j]);
	      //check for exceeding velocity limit;
	      if (l_part[npn].v[j] > vmax[j])
		{
		  if (verbose)
		    printf("Exceeded velocity limit\n");
		  l_part[npn].v[j] = vmax[j];
		}
	      if (l_part[npn].v[j] < (-vmax[j]))
		{
		  if (verbose)
		    printf("Exceeded velocity limit\n");
		  l_part[npn].v[j] = -vmax[j];
		}
	      //move
	      if ((switched == 1) && (npn == 0))
		{
		  switched = 0;
		}
	      else
		{
		  l_part[npn].x[j] += l_part[npn].v[j];
		}
	      //check boundaries and reflect velocity
#ifdef SS
	      if (j == (offset+1)) //phi angle of source position
		{
		  if (l_part[npn].x[j] > params->u[j])
		    {
		      l_part[npn].x[j] -= params->u[j];
		      //		      l_part[npn].v[j] = -l_part[npn].v[j];
		    }
		  if (l_part[npn].x[j] < params->l[j])
		    {
		      l_part[npn].x[j] += params->u[j];
		      //		      l_part[npn].v[j] = -l_part[npn].v[j];
		    }
		}
	      else
		{
#endif
		  if (l_part[npn].x[j] > params->u[j])
		    {
		      l_part[npn].x[j] = params->u[j];
		      l_part[npn].v[j] = -l_part[npn].v[j];
		    }
		  if (l_part[npn].x[j] < params->l[j])
		    {
		      l_part[npn].x[j] = params->l[j];
		      l_part[npn].v[j] = -l_part[npn].v[j];
		    }
#ifdef SS
		}
#endif
	      //move to parameter structure and get likelihood
	      params->values[j] = l_part[npn].x[j];
	      //update the mean value and the memory
	      l_part[npn].mean[j] = ((iter+1) * l_part[npn].mean[j] + l_part[npn].x[j])/(double)(iter+2);
	      for (i = 0; i < (NMEM-1); i++)
		l_part[npn].memory[j][i+1] = l_part[npn].memory[j][i];
	      l_part[npn].memory[j][0] = l_part[npn].x[j];
	    }  
	  calculate_phi(a_ab,phi,*params,Nplsr,pulsars);
	  
	  likelihood = compute_likelihood(phi,pulsars,Nplsr);
	  //  printf("GBEST %d\t%g\t%g\t%g\n",i,parts[i].x[0],parts[i].x[1],likelihood);
	  
	  //check if new personal best
	  if (likelihood < l_part[npn].pmax)
	    {
	      l_part[npn].pmax = likelihood;
	      for (j = 0; j < NCOEFF; j++)
		l_part[npn].pbest[j] = params->values[j];
	    }
	  //          printf("      PBEST ");
	  //  for (j = 0; j < NCOEFF; j++)
	  //    p
	  //set new global for all parts
	} //end per node loop
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Gather(l_part,NPERNODE*sizeof(struct particle),MPI_BYTE,parts,NPERNODE*sizeof(struct particle), MPI_BYTE, 0, MPI_COMM_WORLD);
      if (rank == 0)
	{
	  for (j = 0; j < NCOEFF; j++)
	    {
	      psi[j] = 0.0;
	      bovern[j] = 0.0; W[j] = 0.0;
	    }
	  for (i = 0; i < (NPERNODE*nproc); i++)
	    {
	      if (parts[i].pmax < gmax)
		{
		  gmax = parts[i].pmax;
		  for (j = 0; j < NCOEFF; j++)
		    gbest[j] = parts[i].pbest[j];
		}
	      //update the global mean
	      for (j = 0; j < NCOEFF; j++)
		{
		  psi[j] += parts[i].mean[j]/(double) (nproc*NPERNODE);
		}
	      //print current positions in Agw
	      //	      fprintf(plotfile,"%g\t",parts[i].x[3]);
	    }
	  //calculate B/n value from Rubin criterion
	  for (j = 0; j < NCOEFF; j++)
	    {
	      for (i = 0; i < (NPERNODE*nproc); i++)
		{
		  bovern[j] += 1.0/((double) (NPERNODE*nproc) - 1.0) * (parts[i].mean[j] - psi[j])* (parts[i].mean[j] - psi[j]);
		  for (k = 0; k < intmin((iter+1),NMEM); k++) 
		    W[j] += 1.0/((double) (NPERNODE*nproc) - 1.0) * (parts[i].memory[j][k] - parts[i].mean[j])* (parts[i].memory[j][k] - parts[i].mean[j]);
		}
	      sigma_plus[j] = (double) iter/(iter+1.0) * W[j]  + bovern[j];
	      Rhat[j] = (double) ((NPERNODE*nproc) + 1.0)/(NPERNODE*nproc) * sigma_plus[j]/W[j] - (double) iter/((iter+1)*(NPERNODE*nproc));
	      //	      printf("Rhat %d is %g\n",j,Rhat[j]);
	    }
	  //fprintf(plotfile,"%g\n",Rhat[3]);
	}
      //Bcast gbest. Nodes dont need to know the gmax
      MPI_Bcast(gbest,NCOEFF,MPI_DOUBLE,0,MPI_COMM_WORLD);
      MPI_Bcast(Rhat,NCOEFF,MPI_DOUBLE,0,MPI_COMM_WORLD);
#ifdef SWITCHEROO
#ifdef REDNOISE
#ifndef POWERLAW
      //let's see if some parameters between red noise and GWB can be switched to increase likelihood
      //use NCOEFF/2 nodes for this
      //all nodes know the gbest and should also know the gmax
      int l_toswitch = 0;
      int *g_toswitch;
      if (rank == 0)
	g_toswitch = (int *) malloc(NPERNODE*nproc*sizeof(int));
      if ((iter % 5) == 4)
	{
	  MPI_Bcast(&gmax,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	  if (rank < (NCOEFF/2))
	    {
	      int iswitch = rank;
	      for (j = 0; j < NCOEFF; j++)
		{
		  params->values[j] = gbest[j];
		}
	      params->values[iswitch] = gbest[iswitch+NCOEFF/2];
	      params->values[iswitch+NCOEFF/2] = gbest[iswitch];
	      calculate_phi(a_ab,phi,*params,Nplsr,pulsars);
	      
	      likelihood = compute_likelihood(phi,pulsars,Nplsr);
	      if (likelihood < gmax)
		{
		  printf("On node %d the switch made the likelihood better!  %g\t%g\n",iswitch,gbest[iswitch],gbest[iswitch+NCOEFF/2]);
		  //now i also need to distribute this shit somehow
		  l_toswitch = 1;
		  //just overwrite the local nodes part 0 properties
		  //		  l_part[0].pmax = likelihood;
		  //		  for (j = 0; j < NCOEFF; j++)
		  //		    {
		  //		      l_part[0].pbest[j] = params->values[j];
		  //		      l_part[0].x[j] = params->values[j];
		  //		    }
		}
	      else
		{
		  l_toswitch = 0;
		}
	    }
	  MPI_Gather(&l_toswitch,1,MPI_INT,g_toswitch,1,MPI_INT,0,MPI_COMM_WORLD);
//
//	  MPI_Gather(l_part,NPERNODE*sizeof(struct particle),MPI_BYTE,parts,NPERNODE*sizeof(struct particle), MPI_BYTE, 0, MPI_COMM_WORLD);
	  if (rank == 0)
	    {
	      double temp;
	      for (j = 0; j < (NCOEFF/2); j++)
		{
		  if (g_toswitch[j] == 1)
		    {
		      printf("Switching %d\t%g\t%g\n",j,gbest[j],gbest[j+NCOEFF/2]);
		      temp = gbest[j];
		      gbest[j] = gbest[j+NCOEFF/2];
		      gbest[j+NCOEFF/2] = temp;
		      switched = 1;
		    }
		}
	      if (switched == 1)
		{
		  for (j = 0; j < NCOEFF; j++)
		    {
		      params->values[j] = gbest[j];
		      l_part[0].x[j] = gbest[j];//put it in rank 0 particle
		    }
		  calculate_phi(a_ab,phi,*params,Nplsr,pulsars);
		  
		  likelihood = compute_likelihood(phi,pulsars,Nplsr);
		  printf("\n\n %g ------> %g\n",gmax,likelihood);
		  if (likelihood < gmax)
		    gmax = likelihood;
		  l_part[0].pmax = likelihood;
		  for (j = 0; j < NCOEFF; j++)
		    l_part[0].pbest[j] = params->values[j];
		}
	    }

//
//
//	      for (i = 0; i < (NPERNODE*nproc); i++)
//		{
//		  if (parts[i].pmax < gmax)
//		    {
//		      gmax = parts[i].pmax;
//		      for (j = 0; j < NCOEFF; j++)
//			gbest[j] = parts[i].pbest[j];
//		    }
//		  //print current positions in Agw
//		  //	      fprintf(plotfile,"%g\t",parts[i].x[3]);
//		}
//	      //calculate B/n value from Rubin criterion
//	    }
	  //Bcast gbest. Nodes dont need to know the gmax
	  MPI_Bcast(gbest,NCOEFF,MPI_DOUBLE,0,MPI_COMM_WORLD);
	  
	}
#endif
#endif
#endif
      int breakloop = 1;
      for (j = 0; j < NCOEFF; j++)
	if (fabs(Rhat[j]-1.0) > 0.01)
	  breakloop = 0;
      if (breakloop == 1)
	{
	  if (rank == 0)
	    printf("Rhat criterion reached!\n");
	  break;
	}
      //print best position to output

    } //end iter loop
  double f;
  if (rank == 0)
    {
#ifndef POWERLAW
#ifdef REDNOISE
      for (i = 0; i < (NCOEFF/2); i++)
	{
	  //	  f = (float) (i+1) * ffund;
	  f = params->freqs[i];
	  //	  fprintf(ofile,"%g\t%g\t%g\n",f,pow(10.0,gbest[i]),pow(10.0,gbest[i+NCOEFF/2]));
	}
#else
#ifdef SS
      //      fprintf(ofile,"% 3.3g % 3.3f % 3.3f % 3.3f % 3.3f\n",(params->freqs[params->ifreq]),ss_bayes,gmax,gbest[offset+0],gbest[offset+1]);
      printf("END OF RUN\n\n% 3.3g % 3.3f % 3.3f % 3.3f % 3.3f\n\n",pulsars[0].freqs[NFFT/2-1],ss_bayes,gmax,gbest[offset+0],gbest[offset+1]);
#else
      for (i = 0; i < NCOEFF; i++)
	{
	  f = params->freqs[i];
	  //	  f = (float) (i+1) * ffund;
	  //fprintf(ofile,"%g\t%g\n",f,pow(10.0,gbest[i]));
	}
#endif
#endif
#else
#ifdef REDNOISE
      //      fprintf(ofile,"%g\t%g\t%g\t%g\t%g\n",gbest[0],gbest[1],gbest[2],gbest[3],gmax);
#else
#ifdef SS
      //fprintf(ofile,"% 3.3g % 3.3f % 3.3f % 3.3f % 3.3f % 3.3f % 3.3f % 3.3f\n",(params->freqs[params->ifreq]),ss_bayes,gmax,gbest[offset+0],gbest[offset+1],gbest[offset+2],gbest[offset+4],gbest[offset+5]);
	     printf("END OF RUN\n\n% 3.3g % 3.3f % 3.3f % 3.3f % 3.3f % 3.3f % 3.3f % 3.3f\n\n",pulsars[0].freqs[NFFT/2-1],ss_bayes,gmax,gbest[offset+0],gbest[offset+1],gbest[offset+2],gbest[offset+4],gbest[offset+5]);
#else
      //      fprintf(ofile,"%g\t%g\t%g\n",gbest[0],gbest[1],gmax);
#endif
#endif
#endif
      //      fclose(otfile);
      //fflush(ofile);
    }
}
#endif 

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


void add_signal(struct mypulsar *psr, pulsar * t_psr, struct parameters params, struct source source_pars)
{
  double theta_a = 0.5*PI - psr->dec;
  double phi_a = psr->raj;
  double L_a = 1.0; //distance to the pulsar
  
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
      sn_p = 0.0;
      cs_p = 0.0;
//      sn_p = sin(phase_p)/denom_2;
//      cs_p = cos(phase_p)/denom_2;

     
      //add signal to the residual
      t_psr->obsn[i].sat = (psr->oldbat[i] + (a1*geo.Fac + a2*geo.Fas)*(sn_p - sn) + (a3*geo.Fac + a4*geo.Fas)*(cs_p -cs))/86400.0;
    }

}



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
  //  FILE *ofile;

  //to hold all pulsar information, including individual G matrices
  ObjFunc All;

  //  struct mypulsar *pulsars;
  pulsar * tempo_psrs;
  //  ofile = fopen(argv[1],"w");
  Nplsr = (argc-2);
  int iter = atoi(argv[1]);
  //read filenames
  
  tempo_psrs = (pulsar *) malloc(MAX_PSR*sizeof(pulsar));

  //  struct parameters params;
  //  All.params.omega = 2.0*PI*atof(argv[2]);

  for (i = 0; i < Nplsr; i++)
    {
      //      filenames[i] = (char *) malloc(60*sizeof(char));
      strcpy(pulsarname[i],argv[i+2]);
      sprintf(filenames[i],"%s/%s_mod.tim",pulsarname[i],pulsarname[i]);
      sprintf(parfilenames[i],"%s/%s.par",pulsarname[i],pulsarname[i]);
//      sprintf(filenames[i],"%s.tim",pulsarname[i]);
//      sprintf(parfilenames[i],"%s.par",pulsarname[i]);
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
  //  printf("%g\n",tend-tstart);
  All.pulsars = (struct mypulsar *) malloc(Nplsr * sizeof(struct mypulsar));

  for (i = 0; i < Nplsr; i++)
    {
      strcpy(All.pulsars[i].name,pulsarname[i]);
    }

  All.Nplsr = Nplsr;

  //  gsl_rng *r;
  All.r = gsl_rng_alloc (gsl_rng_mt19937);
  gsl_rng_set(All.r,time(NULL));
  /* ------------------------- */
  tstart = omp_get_wtime();
  initialize_pulsars_fromtempo(tempo_psrs,All.pulsars,Nplsr,&Ndim,&Ntot,&(All.params),&(All.dim),0);
  tend = omp_get_wtime();
  //  printf("init took %f sec\n",tend-tstart);

  All.l = (double *) malloc(All.dim * sizeof(double));
  All.u = (double *) malloc(All.dim * sizeof(double));
  All.x0 = (double *) malloc(All.dim * sizeof(double));
  All.xp0 = (double *) malloc(All.dim * sizeof(double));

  ObjFuncInitLimits(&All);
  ObjFuncRandomizePoint(&All,All.x0);
  ObjFuncRandomizePoint(&All,All.xp0);

  //Ntot = total number of residuals.pu Ndim = reduced number of residuals
  /* ------------------------- */
  int changed = 1;

  for (i = 0; i < Nplsr; i++)
    {
      //      All.pulsars[i].freqs[NFFT/2-1] = All.params.omega/(2.0*PI);//the freq to be investigated
      //All.pulsars[i].index = All.pulsars[i].n_sample-2;
      //here I compute FNF per pulsar
      compute_C_matrix(&(All.pulsars[i]),&(All.params)); //computes FNF and FNT
    }

  FILE * outfile;
  char oname[50];
  double tstart_tot = omp_get_wtime();

  All.params.tspan = 0.0;
  for (i = 0; i < Nplsr; i++)
    if (All.pulsars[i].tspan > All.params.tspan)
      All.params.tspan = All.pulsars[i].tspan;

//  struct my_matrix * a_ab;
//  struct my_matrix * phi;

  All.a_ab = my_matrix_alloc(Nplsr,Nplsr);
#ifdef DM
  All.phi = my_matrix_alloc(Nplsr*NFFT*2,Nplsr*NFFT*2);
#else
  All.phi = my_matrix_alloc(Nplsr*NFFT,Nplsr*NFFT);
#endif
  init_a_ab(All.pulsars,All.a_ab,All.Nplsr);

  double detGNG = 0.0;

  int offset = 2;
  All.params.u[0] = -13.0;
  All.params.l[0] = -17.0;
  All.params.u[1] = 5.0;
  All.params.l[1] = 3.0;
#ifdef SS
  All.params.l[offset+0] = 0.0;//source theta                                                                                                                                
  All.params.u[offset+0] = PI;
  All.params.l[offset+1] = 0.0;//source phi                                                                                                                                  
  All.params.u[offset+1] = 2.0*PI;//2.0*PI-0.02;                                                                                                                             
//  All.params.l[offset+0] = 0.7;//source theta                                                                                                                                
//  All.params.u[offset+0] = 1.3;
//  All.params.l[offset+1] = 1.0;//source phi                                                                                                                                  
//  All.params.u[offset+1] = 2.0;//2.0*PI-0.02;                                                                                                                             
  All.params.l[offset+2] = -20.0;//log of amplitude                                                                                                                         
  All.params.u[offset+2] = -6.0;
#endif
#ifdef ANGLES
  All.params.l[offset+3] = 0.0;//psi
  All.params.u[offset+3] = PI;
  All.params.l[offset+4] = 0.0;//iota                                                                                                                                       
  All.params.u[offset+4] = PI;
  All.params.l[offset+5] = 0.0;//phi0                                                                                                                                     
  All.params.u[offset+5] = 2.0*PI;

#endif

  FILE *fp;

   fp = fopen("twalk2.out", "w");

   TWalk twalk;
   RandomSeed(&RNG, (unsigned long int)53);
   TWalkCTOR(&twalk, &All, All.x0,
             All.xp0, ObjFuncGetDim(&All));
   (void)TWalkSimulation(&twalk, iter, fp, 1, NULL, NULL);

//  struct source source_pars;
//  source_pars.Amp = pow(10.0,-12.5);
//  source_pars.theta_s = 1.0;
//  source_pars.phi_s = 1.5;
//  source_pars.Mc = 1e9;
//  source_pars.psi = 1.0;
//  source_pars.phi0 = 2.5;
//  source_pars.iota = 0.8;
//  source_pars.fr = 5.12691e-08;

  //  print_residuals("PRE",pulsars[2]);
//  for (i = 0; i < Nplsr; i++)
//    add_signal(&(pulsars[i]),&(tempo_psrs[i]),params,source_pars);
//
//  formBatsAll(tempo_psrs,Nplsr);
//  formResiduals(tempo_psrs,Nplsr,0.0);
//  doFitAll(tempo_psrs,Nplsr,0);
//
//  initialize_pulsars_fromtempo(tempo_psrs,pulsars,Nplsr,&Ndim,&Ntot,&params,1);
//
//  for (i = 0; i < Nplsr; i++)
//    {
//      compute_C_matrix(&(pulsars[i]),&params); //computes FNF and FNT
//    }

  //print_residuals("POST",pulsars[2]);

//#ifdef GRID
//  maximize_likelihood(a_ab,phi,detGNG,Nplsr,&params,pulsars);
//#endif
//#ifdef MPI
//  maximize_likelihood_mpi(a_ab,phi,Nplsr,&params, pulsars);
//#endif
  //  initialize_FNF(FNF,FNT,Ninv,&tNt,pulsars,F,Ntot,Nplsr,&detGNG,Ndim);
  

//  //  for (params.omega = 7.*(2.0*PI); params.omega < (1e-07*(2.0*PI)); params.omega += (1e-8*2.0*PI))
//  for (ifreq = 0; ifreq < nfreqs; ifreq++)
//    {
//      params.omega = h_freqs[ifreq]*2.0*PI;
//#else
//
//  int n_per_proc = nfreqs/numproc;
//  printf("Running %d per proc, total %d procs and %d freqs\n",n_per_proc,numproc,nfreqs);
//  for (ifreq = 0; ifreq < n_per_proc; ifreq++)
//    {
//      params.omega = h_freqs[rank*n_per_proc + ifreq] *2.0*PI;
//#endif
//#ifdef F0
//      sprintf(oname,"%6.4e.fp",params.omega/(2.0*PI));
//      outfile = fopen(oname,"w");
//      //      pulsars[0].index = 23;
//      for (i = 0; i < 1000; i++)
//	{
//	  //fill up psrs with noise
//	  //      for (j = 0; j < Nplsr; j++)
//	  double tstart = omp_get_wtime();
//
//	  for (j = 0; j < Nplsr; j++)
//	    {
//	      //	      pulsars[j].index += 2;
//	      pulsars[j].index = gsl_rng_uniform_int(r,pulsars[j].n_sample-1);
//	      compute_C_matrix(&(pulsars[j]),&params);
//	    }
//
//	  double tend = omp_get_wtime();
//	  if (verbose == 3)
//	    printf("compute_C_matrix\t%g\n",tend-tstart);
//	  //	  compute_Nwiggle(&(pulsars[0]));
//	  tstart = omp_get_wtime();
//
//    //    psr->Gres->data[i] = gsl_ran_gaussian(r,1.0);
//	  
//	  for (j = 0; j < Nplsr; j++)
//	    create_noise_only(&(pulsars[j]),r);
//	  tend = omp_get_wtime();
//	  if (verbose == 3)
//	    printf("create_noise \t%g\n",tend-tstart);
//	  tstart = omp_get_wtime();
//	  Fp = compute_Fp(pulsars,&params,Nplsr);
//	  tend = omp_get_wtime();
//	  if (verbose == 3)
//	    printf("compute_Fp \t%g\n",tend-tstart);
//	  fprintf(outfile,"% 6.4e\t% 6.4e\t%d\n",Fp.tCt,Fp.tHt,Fp.used);
//	  fflush(outfile);
//	  //printf("% 6.4e\n",Fp.tHt);
//	}
////      double tend_tot = omp_get_wtime();
////      if (verbose == 3)
////	printf("Duration all evaluation:\t%g\n",tend_tot-tstart_tot);
//      fclose(outfile);
//#else
//      Fp = compute_Fp(pulsars,&params,Nplsr);
//      changed = 0;
//      fprintf(ofile,"% 6.4e  % 6.4e  % 6.4e %d\n",params.omega,Fp.tCt,Fp.tHt,Fp.used);
//#endif
//    }
//  double tend_tot = omp_get_wtime();
//  fprintf(stderr,"Duration all evaluation:\t%g\n",tend_tot-tstart_tot);
//#endif //else UPPER
#ifdef CULA
  culaShutdown();
#endif
#ifdef MPI
  if(rank ==0)
#endif
    //    fclose(ofile);
#ifdef MPI
  MPI_Finalize();
#endif
  return 0;
}
