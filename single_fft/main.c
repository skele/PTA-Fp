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

#define verbose 2
#define NFFT 24
#define N_SAMPLE_MAX 27000
#define MAX_PSR 45
#define MAX_BE 20

#define ANGLES

#include <mydefs.h>

//#define UPPER
//#define EFAC
//#define DM

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
#define GRID

//#define PCA

//#define SINGLE
//#define RUNITY
#define SS
//#define REDNOISE
#define POWERLAW
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
      //compute design matrix here
      if (only_res == 0)
	{
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
	  pulsars[i].FNT = my_vector_alloc(2*NFFT);
#else
	  pulsars[i].phi_inv = my_matrix_alloc(NFFT,NFFT);
	  pulsars[i].F = my_matrix_alloc(pulsars[i].N,NFFT);
	  pulsars[i].GF = my_matrix_alloc(pulsars[i].N_m,NFFT);
	  pulsars[i].FNT = my_vector_alloc(NFFT);
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

#ifdef EFAC
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
#endif
	  //      compute_Nwiggle(&(pulsars[i]));
      
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
      temp = (tmax - tmin);
      //temp = *tspan*1.306;                                                                                                                                                 
      //  temp = *tspan;
      //temp = *tspan*1.18;                                                                                                                                                  
      double ffund = 1.0/(1.0*temp);
      //set up frequency grid                                                                                                                                                
      for (a = 0; a < Nplsr; a++)
	{
	  for (b = 0; b < (NFFT/2); b++)
	    {
	      pulsars[a].freqs[b] = ffund*(b+1);
	      //      printf("%d\t%e\n",i,par->freqs[i]);
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
  psr->FNF = my_matrix_alloc(2*NFFT,2*NFFT);
#else
  FN = my_matrix_alloc(NFFT,psr->N_m);
  psr->FNF = my_matrix_alloc(NFFT,NFFT);
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
#ifdef REDNOISE
  offset = 4;
#endif
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

  //Compute k,u,v vectors                                                                                                                                                
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
  my_matrix_set_zero(phi);
#ifdef POWERLAW
  double Agw = pow(10.0,params.values[0]);
  double fac = Agw * Agw / (12.0*PI*PI) * 3.16e22;
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
#ifdef POWERLAW
              //f = (float) (i+1) * ffund;                                                                                                                               
              f = pulsars[a].freqs[i];
              power =  a_ab->data[b*a_ab->m + a]*fac* pow(f/3.17e-08,-params.values[1])/params.tspan;
#else
              power =  a_ab->data[b*a_ab->m + a]*pow(10.0,params.values[i]);
#endif
              //printf("PHI\t%g\t%g\n",f,fac* pow(f/3.17e-08,-4.333) /params.tspan);                                                                                            
              phi->data[(ind_c+2*i)*phi->m + ind_r + 2*i] = power;
              phi->data[(ind_c+2*i+1)*phi->m + ind_r + 2*i + 1] = power;
#ifdef SS
              if (i == (NFFT/2-1))
                {
                  g_fac = geo_fac(&params,pulsars,a,b);
                  phi->data[(ind_c+2*i)*phi->m + ind_r + 2*i] +=  g_fac.Fac;
                  phi->data[(ind_c+2*i+1)*phi->m + ind_r + 2*i + 1] += g_fac.Fas;
                }
#endif

#ifdef REDNOISE
              if (a == b)
                {
		  double rAgw = pulsars[a].rA;
		  double rgamma = pulsars[a].rgamma;
		  double rfac = rAgw * rAgw / (12.0*PI*PI) * 3.16e22;
#ifdef DM
		  double dmAgw = pulsars[a].dmA;
		  double dmgamma = pulsars[a].dmgamma;
		  double dmfac = dmAgw * dmAgw / (12.0*PI*PI) * 3.16e22;
#endif

#ifdef POWERLAW
                  power = rfac* pow(f/3.17e-08,-params.values[3])/params.tspan;
#else
                  //add red noise for diagonal elements                                                                                                                  
                  power = pow(10.0,params.values[i+NFFT/2]);
#endif
                  phi->data[(ind_c+2*i)*phi->m + ind_r + 2*i] += power;
                  phi->data[(ind_c+2*i+1)*phi->m + ind_r + 2*i + 1] += power;
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

double compute_likelihood(struct my_matrix * phi,struct mypulsar * pulsars, int Nplsr)
{
  double likeli = 0.0;
  double tNt = 0.0;
  int i;
  //get inverse of phi                                                                                                                                                   
  struct my_matrix * phiinv;
  phiinv = my_matrix_alloc(NFFT*Nplsr,NFFT*Nplsr);
  struct my_matrix * sigmainv;
  sigmainv = my_matrix_alloc(NFFT*Nplsr,NFFT*Nplsr);
  struct my_matrix * cholesky_phi;
  cholesky_phi = my_matrix_alloc(NFFT*Nplsr,NFFT*Nplsr);
  struct my_matrix * cholesky_sigma;
  cholesky_sigma = my_matrix_alloc(NFFT*Nplsr,NFFT*Nplsr);
  my_matrix_memcpy(phiinv,phi);
  //my_matrix_memcpy(sigmainv,FNF);
  
  double  detPhi = 0.0;
  double detSigma = 0.0;

  my_matrix_print("phi\0",phi);

//  if (get_inverse_cholesky(phiinv,cholesky_phi,NFFT*Nplsr) == 0 && (verbose))                                                                                          
//    printf("My inversion of phi work\n");                                                                                                                              
  if (get_inverse_lu(phiinv,cholesky_phi,NFFT*Nplsr,&detPhi) == 0 && (verbose))
    printf("My inversion of phi work\n");
  //construct phiinv + FNF object
  int a,b,j;
  int ind_c = 0,ind_r = 0;
  my_matrix_memcpy(sigmainv,phiinv);
  for (a = 0; a < Nplsr; a++)
    {
      for (i = 0; i < NFFT; i++)
	for (j = 0; j < NFFT; j++)
	  sigmainv->data[(i + ind_c)*sigmainv->m + ind_c + j] = pulsars[a].FNF->data[i*pulsars[a].FNF->m + j];
      ind_c += NFFT;
    }
  //  my_matrix_print("phi\0",sigmainv);

  //invert Sigma
  if (get_inverse_lu(sigmainv,cholesky_sigma,NFFT*Nplsr,&detSigma) == 0 && (verbose))
    printf("My inversion of sigma worked\n");
  struct my_vector * dsigma;
  dsigma = my_vector_alloc(NFFT*Nplsr);
  struct my_vector * FNTbig;
  FNTbig = my_vector_alloc(NFFT*Nplsr);
  for (a = 0; a < Nplsr; a++)
    for (i = 0; i < NFFT; i++)
      FNTbig->data[a*NFFT + i] = pulsars[a].FNT->data[i];
  double dsd;
  //combine FNT into FNTbig vector
  my_dsymv(1.0,sigmainv,FNTbig,0.0,dsigma); 
  my_vector_mult(FNTbig,dsigma,&dsd);
  //add together tNt per pulsar
  double detN = 0.0;
  for (i = 0; i < Nplsr; i++)
    {
      //tNt += pulsars[i].tNt;
      //    detN += pulsars[i].det;
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

  params->values[0] = -14.0;
  params->values[1] = 4.333;
  int offset = 2;

#ifndef SS
  for (params->values[2] = params->l[2]; params->values[2] < params->u[2]; params->values[2] += (params->u[2]-params->l[2])/20.0)
    for (params->values[3] = params->l[3]; params->values[3] < params->u[3]; params->values[3] += (params->u[3]-params->l[3])/20.0)
      {
        calculate_phi(a_ab,phi,*params,Nplsr,tspan,pulsars);

        likeli_old = compute_likelihood(phi,pulsars,Nplsr);
        //printf("%f\t%f\t%f\n",params->values[0],params->values[1],likeli_old);                                                                                         
	printf("%f\t%f\t%f\n",params->values[2],params->values[3],likeli_old);
	fflush(stdout);
      }
#else
#ifdef REDNOISE
  offset = 4;
  params->values[2] = -14.0;
  params->values[3] = 2.0;
#endif

  //#else                                                                                                                                                                
  params->values[offset+0] = 1.0;
  params->values[offset+1] = 1.5;
  params->values[offset+2] = -8.0;
  params->values[offset+3] = 1.0;
  params->values[offset+4] = 0.8;
  params->values[offset+5] = 2.5;

  calculate_phi(a_ab,phi,*params,Nplsr,pulsars);
  likeli_old = compute_likelihood(phi,pulsars,Nplsr);
  printf("#VALUE AT %f\t%f\t%f\n",params->values[6],params->values[7],likeli_old);
  double minimum = 1e10;
  for (params->values[offset+2] = params->l[offset+2]; params->values[offset+2] < params->u[offset+2]; params->values[offset+2] += (params->u[offset+2]-params->l[offset+2])/150.0)
    //for (params->values[offset+0] = params->l[offset+0]; params->values[offset+0] < params->u[offset+0]; params->values[offset+0] += (params->u[offset+0]-params->l[offset+0])/30.0)
    //for (params->values[offset+1] = params->l[offset+1]; params->values[offset+1] < params->u[offset+1]; params->values[offset+1] += (params->u[offset+1]-params->l[offset+1])/30.0)
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
        //printf("%f\t%f\t%f\t%f\n",params->values[offset-2+5],params->values[offset-2+6],params->values[offset-2+3],likeli_old);                                        
	//printf("%f\t%f\t%f\n",params->values[offset+0],params->values[offset+1],likeli_old);
        printf("%f\t%f\n",params->values[offset+2],likeli_old);                                                                                                        
        fflush(stdout);
      }

#endif


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
      sn_p = sin(phase_p)/denom_2;
      cs_p = cos(phase_p)/denom_2;

     
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
  FILE *ofile;

  //to hold all pulsar information, including individual G matrices
  struct mypulsar *pulsars;
  pulsar * tempo_psrs;
  ofile = fopen(argv[1],"w");
  Nplsr = (argc-3);
  //read filenames
  
  tempo_psrs = (pulsar *) malloc(MAX_PSR*sizeof(pulsar));

  struct parameters params;
  params.omega = 2.0*PI*atof(argv[2]);

  for (i = 0; i < Nplsr; i++)
    {
      //      filenames[i] = (char *) malloc(60*sizeof(char));
      strcpy(pulsarname[i],argv[i+3]);
      sprintf(filenames[i],"%s.tim",pulsarname[i]);
      sprintf(parfilenames[i],"%s.par",pulsarname[i]);
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
  pulsars = (struct mypulsar *) malloc(Nplsr * sizeof(struct mypulsar));

  for (i = 0; i < Nplsr; i++)
    {
      strcpy(pulsars[i].name,pulsarname[i]);
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

  for (i = 0; i < Nplsr; i++)
    {
      pulsars[i].freqs[NFFT/2-1] = params.omega/(2.0*PI);//the freq to be investigated
      pulsars[i].index = pulsars[i].n_sample-2;
      //here I compute FNF per pulsar
      compute_C_matrix(&(pulsars[i]),&params); //computes FNF and FNT
    }

  FILE * outfile;
  char oname[50];
  double tstart_tot = omp_get_wtime();

  params.tspan = 0.0;
  for (i = 0; i < Nplsr; i++)
    if (pulsars[i].tspan > params.tspan)
      params.tspan = pulsars[i].tspan;

  struct my_matrix * a_ab;
  struct my_matrix * phi;

  a_ab = my_matrix_alloc(Nplsr,Nplsr);
#ifdef DM
  phi = my_matrix_alloc(Nplsr*NFFT*2,Nplsr*NFFT*2);
#else
  phi = my_matrix_alloc(Nplsr*NFFT,Nplsr*NFFT);
#endif
  init_a_ab(pulsars,a_ab,Nplsr);

  double detGNG = 0.0;

  int offset = 2;
  params.l[offset+0] = 0.0;//source theta                                                                                                                                
  params.u[offset+0] = PI;
  params.l[offset+1] = 0.0;//source phi                                                                                                                                  
  params.u[offset+1] = 2.0*PI;//2.0*PI-0.02;                                                                                                                             
  params.l[offset+2] = -20.0;//log of amplitude                                                                                                                          
  params.u[offset+2] = -5.0;

  
  struct source source_pars;
  source_pars.Amp = pow(10.0,-30.0);
  source_pars.theta_s = 1.0;
  source_pars.phi_s = 1.5;
  source_pars.Mc = 1e9;
  source_pars.psi = 1.0;
  source_pars.phi0 = 2.5;
  source_pars.iota = 0.8;
  source_pars.fr = 2.e-08;

  for (i = 0; i < Nplsr; i++)
    add_signal(&(pulsars[i]),&(tempo_psrs[i]),params,source_pars);

  formBatsAll(tempo_psrs,Nplsr);
  formResiduals(tempo_psrs,Nplsr,0.0);
  doFitAll(tempo_psrs,Nplsr,0);

  initialize_pulsars_fromtempo(tempo_psrs,pulsars,Nplsr,&Ndim,&Ntot,&params,1);

  maximize_likelihood(a_ab,phi,detGNG,Nplsr,&params,pulsars);

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
    fclose(ofile);
#ifdef MPI
  MPI_Finalize();
#endif
  return 0;
}
