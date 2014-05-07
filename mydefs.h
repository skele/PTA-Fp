//#include <gsl/gsl_matrix.h>
//#include <gsl/gsl_math.h>

#define NCOEFF 2
#define NMEM 20

#define THRESH 10000

#define HAVEDIST 27

char distance_keys[HAVEDIST][11] = {"J0030+0451\0","J0610-2100\0","J0613-0200\0","J0751+1807\0","J0900-3144\0","J1012+5307\0","J1024-0719\0","J1455-3330\0","J1600-3053\0","J1640+2224\0","J1643-1224\0","J1713+0747\0","J1730-2304\0","J1738+0333\0","J1744-1134\0","J1804-2717\0","J1857+0943\0","J1909-3744\0","J1910+1256\0","J1918-0642\0","J1939+2134\0","J2010-1323\0","J2019+2425\0","J2124-3358\0","J2145-0750\0","J2317+1439\0","J2322+2057"} ;
double distance_values[HAVEDIST] = {0.28,5.64,0.9,0.4,0.82,0.7,0.49,0.74,2.4,1.19,0.42,1.05,0.51,1.97,0.42,1.17,0.9,1.26,1.95,1.4,5.0,1.29,0.91,0.3,0.57,1.89,0.78};

enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102 };

enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113,
                         AtlasConj=114};
enum CBLAS_UPLO  {CblasUpper=121, CblasLower=122};
enum CBLAS_SIDE  {CblasLeft=141, CblasRight=142};

struct geo_par
{
  double Fac,Fas;
};

struct source
{
  double theta_s,phi_s,Mc,fr,psi,phi0,Amp,iota;
};

struct Fp
{
  double tCt,tHt;
  int used;
};

struct particle
{
  double x[NCOEFF],v[NCOEFF],pbest[NCOEFF],gbest[NCOEFF],pmax,gmax;
  double mean[NCOEFF];
  double memory[NCOEFF][NMEM];
};

struct my_vector
{
  double *data;
  int n;
};

struct my_matrix
{
  double *data;
  char ulo;
  int m,n;
};

struct mypulsar
{
  char name[50];
  double raj,dec,tspan,DMtspan,det,dist;
  int N,N_m,index;
  int n_be,n_sample;
  int * backends; //points to the first toa index for each backend
  //  double 
  double freqs[NFFT/2];
  double DMfreqs[NFFT/2];
  double *sigma,*oldbat;
  double rA,rgamma;
  double dmA,dmgamma;
  double tNt;
  struct my_matrix *G,*CWN,*GNGinv,*F,*H,*C,*Cinv,*L,*FNF;
  struct my_matrix *GF,*GH,*sample;
  struct my_vector *toa,*res,*Gres,*phi_inv,*obsfreqs,*FNT;
};

struct parameters
{
  double tspan,DMtspan;
  double omega;
  double values[NCOEFF];
  double tNt;
  //  double Agw, gamma_gw, fL;
  double l[NCOEFF];
  double u[NCOEFF];//double bound_Agw[2],bound_gamma_gw[2];
  int * indices;
};

void print_residuals(char * prefix, struct mypulsar psr)
{
  int i;
  for (i = 0; i < psr.N; i++)
    {
      printf("%s %s %g  %g\n",prefix,psr.name,psr.toa->data[i],psr.res->data[i]);
    }
}


extern void dtrmv_(char *uplo,char *TA,char *diag,int *n,double *a,int *lda,double * b,int * incx);

//#ifndef CULA
extern void dpptrf_(char *uplo, int *n, double *A, int *info);
extern void dpotri_(char *uplo, int *n, double *A, int *lda, int *info);
//extern void dgemm_(const char *transa, const char *transb, const int *m,
//		   const int *n, const int *k, const double *alpha, const double *a,
//		   const int *lda, const double *b, const int *ldb, const double *beta,
//		   double *c, const int *ldc);
extern void dgemm_(char *trans, char * transb, int *m, int*n, int*k, double * alpha, double *A, int *lda, double *B, int * ldb, double *beta, double *C, int * ldc);
extern void dsymm_(char *side, char *uplo, int *m, int *n, double *alpha, double *a, int *lda, double *b, int *ldb, double *beta, double *c, int *ldc);
extern void dsymv_(char *uplo, int *n, double *alpha,  double *a, int *lda, double *x, int *incx, double *beta, double *y, int *incy);
extern void dgemv_(char *trans, int *m, int *n, double * alpha, double *a, int *lda, double *x, int *incx, double *beta, double *y, int *incy);
extern void dgesvd_(char *jobu, char *jobvt,int *m, int *n, double A[], int *lda, double s[],
		    double U[], int *ldu, double VT[], int *ldtv, double *work, int * lwork, int*info);
extern void dgetrf_(int *m, int*n, double * a, int*lda, int*ipiv, int*info);
extern void dgetri_(int*n, double * a, int*lda, int*ipiv, double *work, int *lwork, int*info);
extern void dtrtri_(char *uplo, char *diag, int*n, double*a, int *lda, int*info);
//#endif

//this sorts an array of strings, returning an int array with the sorted arguments indices
void argsort(char instrings[MAX_BE][MAX_FLAG_LEN], int sorted[], int n)
{
  int i,j,k;
  int *skip;
  skip = (int*)malloc(n * sizeof(int));
  //search minimum and take it out
  for (i = 0; i < n; i++)
    skip[i] = -1;
  int compare;
  for (i = 0; i < n; i++)
    {
      int min = 0;
      int minindex = -1;
      for (j = 0; j < n; j++)
	{
	  //see if this one is already found
	  int skipit = 0;
	  for (k = 0; k < n; k++)
	    {
	      if (skip[k] == j)
		skipit = 1;
	    }
	  if (skipit == 0)
	    {
	      if (minindex == -1)
		//first element, assume min
		minindex = j;
	      else
		{
		  //compare to minindex one
		  compare = strcmp(instrings[j],instrings[minindex]);
		  if (compare < 0)
		    minindex = j;
		}
	    }
	}
      //now i have the minimun, skip it
      skip[i] = minindex;
      sorted[i] = minindex;
    }
  free(skip);
}

double min(double a, double b)
{
  return (a<b)?a:b;
}
int intmin(int a, int b)
{
  return (a<b)?a:b;
}
double max(double a, double b)
{
  return (a>b)?a:b;
}

void my_matrix_print_pretty(char * header, struct my_matrix * a)
{
  int i,j;
  printf("%10s\n",header);
  for (i = 0; i < a->m; i++)
    {
      for (j = 0; j < a->n; j++)
        printf("%4.2f\t",a->data[j*a->m + i]);
      printf("\n");
    }
  printf("\n");
}

void my_matrix_print(char * header, struct my_matrix * a)
{
  int i,j;
  for (i = 0; i < a->n; i++)
    {
      for (j = 0; j < a->m; j++)
	printf("%10s\t%5d %5d %.2g\n",header,i,j,a->data[i*a->m + j]);
      printf("\n");
    }
  printf("\n");
}

void my_vector_mult(struct my_vector * a, struct my_vector *b, double * result)
{
  double temp = 0;
  int i;
  for (i = 0; i < a->n; i++)
    temp += a->data[i]*b->data[i];
  *result = temp;
}

int my_matrix_add_diagonal(struct my_matrix * a, struct my_vector *b)
{
  int j;
  if ((a->m != b->n) || (a->n != b->n))
    {
      printf("matrix add error: dimensions not the same\n");
      return 1;
    }
  for (j = 0; j < a->n; j++)
    a->data[j*a->m + j] += b->data[j];
  return 0;

}

int my_matrix_add(struct my_matrix * a, struct my_matrix * b)
{
  int i,j;
  if ((a->m != b->m) || (a->n != b->n))
    {
      printf("matrix add error: dimensions not the same\n");
      return 1;
    }
  for (j = 0; j < a->n; j++)
    for (i= 0; i < a->m; i++)
      a->data[j*a->m + i] += b->data[j*b->m + i];
  return 0;
}

int my_matrix_sub(struct my_matrix * a, struct my_matrix * b)
{
  int i,j;
  if ((a->m != b->m) || (a->n != b->n))
    {
      printf("matrix sub error: dimensions not the same\n");
      return 1;
    }
  for (j = 0; j < a->n; j++)
    for (i= 0; i < a->m; i++)
      b->data[j*b->m + i] = a->data[j*a->m + i] - b->data[j*b->m + i];
  return 0;
}

void my_matrix_memcpy(struct my_matrix * a, struct my_matrix * b)
{
  memcpy(a->data,b->data,a->n*a->m*sizeof(double));
}

void my_vector_init(struct my_vector * vec, int n)
{
  vec = (struct my_vector *) malloc (sizeof(struct my_vector));
  vec->data = (double *) malloc(n*sizeof(double));
  vec->n = n;
}

void my_vector_print(struct my_vector * vec)
{
  int i;
  for (i = 0; i < vec->n; i++)
    printf("%d\t%g\n",i,vec->data[i]);
}

struct my_vector * my_vector_alloc(int n)
{
  struct my_vector * vec;
  vec = (struct my_vector *) malloc(sizeof(struct my_vector));
  vec->data = (double *) malloc(n*sizeof(double));
  memset(vec->data,0.0,n*sizeof(double));
  vec->n = n;
  return vec;
}

struct my_vector * my_vector_alloc_count(int *size, int n)
{
  struct my_vector * vec;
  vec = (struct my_vector *) malloc(sizeof(struct my_vector));
  vec->data = (double *) malloc(n*sizeof(double));
  memset(vec->data,0.0,n*sizeof(double));
  vec->n = n;

  *size += n*sizeof(double);

  return vec;
}

struct my_matrix * my_matrix_alloc( int m, int n)
{
  struct my_matrix * mat;
  mat = (struct my_matrix *) malloc (sizeof(struct my_matrix));
  mat->data = (double *) malloc(n*m*sizeof(double));
  if (mat->data == NULL)
    {
      fprintf(stderr,"Matrix cannot be allocated, size %d\n",n*m*sizeof(double));
      return NULL;
    }
  memset(mat->data,0.0,n*m*sizeof(double));
  mat->n = n;
  mat->m = m;
  return mat;
}

struct my_matrix * my_matrix_alloc_count(int *size, int m, int n)
{
  struct my_matrix * mat;
  mat = (struct my_matrix *) malloc (sizeof(struct my_matrix));
  mat->data = (double *) malloc(n*m*sizeof(double));
  if (mat->data == NULL)
    {
      fprintf(stderr,"Matrix cannot be allocated, size %d\n",n*m*sizeof(double));
      return NULL;
    }
  memset(mat->data,0.0,n*m*sizeof(double));
  mat->n = n;
  mat->m = m;
  *size += n*m*sizeof(double);

  if (verbose)
    printf("Allocated a total of %8.3f Mbyte\n",*size/1e6);

  return mat;
}

void my_matrix_init(struct my_matrix * mat, int m, int n)
{
  mat = (struct my_matrix *) malloc (sizeof(struct my_matrix));
  mat->data = (double *) malloc(n*m*sizeof(double));
  mat->n = n;
  mat->m = m;
}

void my_matrix_set_zero(struct my_matrix *mat)
{
  memset(mat->data,0.0,mat->n*mat->m*sizeof(double));
}

void my_matrix_cinit(struct my_matrix * mat, int m, int n)
{
  mat = (struct my_matrix *) malloc (sizeof(struct my_matrix));
  mat->data = (double *) malloc(n*m*sizeof(double));
  mat->n = n;
  mat->m = m;
  my_matrix_set_zero(mat);
}

void my_matrix_free(struct my_matrix * mat)
{
  free(mat->data);
  free(mat);
}
void my_vector_free(struct my_vector * vec)
{
  free(vec->data);
  free(vec);
}

int my_geev(struct my_matrix *A, struct my_matrix *evec, struct my_vector * eval)
{
  int lda;
  char jobvl = 'N';
  char jobvr = 'V';//compute right eigenvectors
  if (A->m != A->n)
    {
      printf("Cannot compute eigenvectors of non square matrix!\n");
      return -1;
    }
  int n = A->m;
  double *imparts;
  imparts = (double*)malloc(n*sizeof(double));
#ifdef CULA
  culaStatus s;
  s = culaDgeev(jobvl,jobvr,n,A->data,lda,eval->data,imparts,imparts,lda,evec->data,lda);
  if(s != culaNoError)
    {
      printf("%s\n", culaGetStatusString(s));
      fprintf(stdout,"****Error in SVD\t%d\n",culaGetErrorInfo(s));
      /* ... Error Handling ... */
    }
#endif
  return 0;
  
}

int get_inverse_tri(struct my_matrix * A)
{
  int lda;
  lda = A->m;
  char uplo = 'L';
  char diag = 'N';
  int n = lda;
#ifdef CULA
  culaStatus s;
  s = culaDtrtri(uplo,diag,n,A->data,lda);
  if(s != culaNoError)
    {
      printf("%s\n", culaGetStatusString(s));
      fprintf(stdout,"****Error in inv tri\t%d\n",culaGetErrorInfo(s));
      /* ... Error Handling ... */
    }
#else
  int info;
  dtrtri_(&uplo,&diag,&n,A->data,&lda,&info);
#endif
  return 0;

}

int my_svd(struct my_matrix * A, struct my_matrix * U, struct my_vector *singular)
{
  int lda;
  lda = A->m;
  char jobu = 'A';
  char jobvt = 'N';
  double dummy[1];
  int ldu = U->m;
#ifdef CULA
  culaStatus s;
  s = culaDgesvd(jobu,jobvt,A->m,A->n,A->data,lda,singular->data,U->data,ldu,dummy,ldu);
  if(s != culaNoError)
    {
      printf("%s\n", culaGetStatusString(s));
      fprintf(stdout,"****Error in SVD\t%d\n",culaGetErrorInfo(s));
      /* ... Error Handling ... */
    }
#else
  int info;
  double * work;
  int lwork = A->m*5;
  work = (double *) malloc(lwork*sizeof(double));
  dgesvd_(&jobu,&jobvt,&(A->m),&(A->n),A->data,&lda,singular->data,U->data,&ldu,dummy,&ldu,work,&lwork,&info);
  free(work);
#endif
  return 0;
}

int my_dtrmv(int TransA, struct my_matrix *A, struct my_vector * B)
{
  char uplo = 'L';
  char TA;
  int lda = A->m;
  if (TransA == CblasTrans)
    {
      TA = 'T';
    }
  else
    {
      TA = 'N';
    }
  char diag = 'N';
  int incx = 1;
  int n = B->n;
  dtrmv_(&uplo,&TA,&diag,&n,A->data,&lda,B->data,&incx);
  return 0;
  
}

int my_dsymv(double alpha, struct my_matrix * A, struct my_vector * B, double beta, struct my_vector * result)
{
  int incx,incy,n,lda;
  incx = 1; incy = 1;
  lda = A->m;
  n = B->n;
#ifdef CULA
  if ((A->m*A->n) > THRESH)
    {
      //      printf("Moving to GPU %d\n",A->m*A->n);
      //#ifdef CULA
      culaStatus s;
      s = culaDsymv('L',n,alpha,A->data,lda,B->data,incx,beta,result->data,incy);
      if(s != culaNoError)
	{
	  printf("%s\n", culaGetStatusString(s));
	  fprintf(stdout,"****Error in Multiplication of sym matrix with vector%d\n",culaGetErrorInfo(s));
	  /* ... Error Handling ... */
	}
    }
else
#endif
      //#else
  {
    char uplo;
    uplo = 'L';
    dsymv_(&uplo,&n,&alpha,A->data,&lda,B->data,&incx,&beta,result->data,&incy);
  }
//#endif
  return 0;
}

int  my_dgemv(int TransA,double alpha,struct my_matrix * A,struct my_vector * B,double beta,struct my_vector * result)
{
  char TA;
  int m,n,lda,incx,incy;
  incx = 1;
  incy = 1;
  m = A->m;
  n = A->n;
  lda = m;
  if (TransA == CblasTrans)
    {
      TA = 'T';
      if (A->m != B->n)
	{
	  printf("A->m != B->n %d\t%d\n",A->m,B->n);
	}
      if (A->n != result->n)
	{
	  printf("A->m != result->n %d\t%d\n",A->n,result->n);
	}
    }
  else
    {
      TA = 'N';
      if (A->n != B->n)
	{
	  printf("A->n != B->n %d\t%d\n",A->n,B->n);
	}
      if (A->m != result->n)
	{
	  printf("A->n != result->n %d\t%d\n",A->m,result->n);
	}
    }
#ifdef CULA
  if ((A->m*A->n) > THRESH)
    {
      //printf("Moving to GPU %d\n",A->m*A->n);
      //#ifdef CULA
      culaStatus s;
      s = culaDgemv(TA,m,n,alpha,A->data,lda,B->data,incx,beta,result->data,incy);
      if(s != culaNoError)
	{
	  printf("%s\n", culaGetStatusString(s));
	  fprintf(stdout,"****Error in Multiplication of matrix with vector%d\n",culaGetErrorInfo(s));
	  /* ... Error Handling ... */
	}
    }
  else
#endif
    //#else
    dgemv_(&TA,&m,&n,&alpha,A->data,&lda,B->data,&incx,&beta,result->data,&incy);
  //  cblas_dgemv(CblasColMajor,TransA,m,n,alpha,A->data,lda,B->data,incx,beta,result->data,incy);
  //#endif
  return 0;
}

int my_dsymm(int side, double alpha, const struct my_matrix * A, const struct my_matrix * B, double beta, struct my_matrix * C)
{
  if (A->m != A->n)
    {
      printf("Not a square matrix in A!\n");
      return -1;
    }
  char cside,uplo;
  if (side == CblasLeft)
    {
      cside = 'L';
      if (A->n != B->m)
	{
	  printf("dsymm A->n != B->m  %d\t%d\n",A->n,B->m);
	}
      if (A->m != C->m)
	{
	  printf("dsymm A->m != C->m  %d\t%d\n",A->m,C->m);
	}
      if (B->n != C->n)
	{
	  printf("dsymm B->n != C->n  %d\t%d\n",B->n,C->n);
	}
    }
  else
    {
      cside = 'R';
      if (B->n != A->m)
	{
	  printf("dsymm A->n != B->m  %d\t%d\n",B->n,A->m);
	}
      if (B->m != C->m)
	{
	  printf("dsymm A->m != C->m  %d\t%d\n",B->m,C->m);
	}
      if (A->n != C->n)
	{
	  printf("dsymm B->n != C->n  %d\t%d\n",A->n,C->n);
	}
    }
  uplo = 'L';
  int m,n,lda,ldb,ldc;
  m = C->m;
  n = C->n;
  lda = A->m;
  ldb = B->m;
  ldc = C->m;
#ifdef CULA
  if ((A->m*A->n) > THRESH)
    {
      //printf("Moving to GPU %d\n",A->m*A->n);
      //#ifdef CULA
      culaStatus s;
      s = culaDsymm(cside,uplo,m,n,alpha,A->data,lda,B->data,ldb,beta,C->data,ldc);
      if(s != culaNoError)
	{
	  printf("%s\n", culaGetStatusString(s));
	  fprintf(stdout,"****Error in symmetric Multiplication %d\n",culaGetErrorInfo(s));
	  /* ... Error Handling ... */
	}
    }
  else
#endif
    //#else
    //  cblas_dsymm(CblasColMajor,side,CblasLower,m,n,alpha,A->data,lda,B->data,ldb,beta,C->data,ldc);
    dsymm_(&cside,&uplo,&m,&n,&alpha,A->data,&lda,B->data,&ldb,&beta,C->data,&ldc);
  //#endif
  return 0;
}

//My wrapper to preform a matrix matrix multiplication of gsl matrixes with cula
//int my_dgemm(int TransA, int TransB, double alpha, const gsl_matrix * A, const gsl_matrix * B, double beta, gsl_matrix * C)
int my_dgemm(int TransA, int TransB, double alpha, const struct my_matrix * A, const struct my_matrix * B, double beta, struct my_matrix * C)
{  
#ifdef CULA
  culaStatus s;
#endif
  //error checking in sizes
  if ((TransA == CblasNoTrans) && (TransB == CblasNoTrans))
    {
      if (A->n != B->m)
	printf("Sizes don't match: dgemm NoTrans NoTrans\t%d\t%d\n",A->n,B->m);
    }
  if ((TransA == CblasTrans) && (TransB == CblasNoTrans))
    {
      if (A->m != B->m)
	printf("Sizes don't match: dgemm Trans NoTrans\t%d\t%d\n",A->m,B->m);
    }
  if ((TransA == CblasNoTrans) && (TransB == CblasTrans))
    {
      if (A->n != B->n)
	printf("Sizes don't match: dgemm NoTrans Trans\t%d\t%d\n",A->n,B->n);
    }
  if ((TransA == CblasTrans) && (TransB == CblasTrans))
    {
      if (A->m != B->m)
	printf("Sizes don't match: dgemm Trans Trans\t%d\t%d\n",A->m,B->m);
    }
  if (TransA == CblasNoTrans)
    {
      if (A->m != C->m)
	printf("Sizes don't match: dgemm A->m and C->m\t%d\t%d\n",A->m,C->m);	
    }
  if (TransA == CblasTrans)
    {
      if (A->n != C->m)
	printf("Sizes don't match: dgemm A->n and C->m\t%d\t%d\n",A->n,C->m);	
    }
  if (TransB == CblasNoTrans)
    {
      if (B->n != C->n)
	printf("Sizes don't match: dgemm B->n and C->n\t%d\t%d\n",B->n,C->n);	
    }
  if (TransB == CblasTrans)
    {
      if (B->m != C->n)
	printf("Sizes don't match: dgemm B->m and C->n\t%d\t%d\n",B->m,C->n);	
    }
  //prepare TransA and TransB
  char TA,TB;
  //prepare dimensions
  int m,n,k,lda,ldb,ldc;
  m = C->m;
  n = C->n;
  if (TransA == CblasTrans)
    {
      TA = 'T';
      k = A->m;
      lda = A->m;
    }
  else
    {
      TA = 'N';
      k = A->n;
      lda = A->m;
    }
  if (TransB == CblasTrans)
    {
      TB = 'T';
      ldb = B->m;
    }     
  else
    {
      TB = 'N';
      ldb = B->m;
    }
  ldc = m;
//  double * cA, *cB, *cC;
//  cA = (double *) malloc(A->size1*A->size2*sizeof(double));
//  cB = (double *) malloc(B->size1*B->size2*sizeof(double));
//  cC = (double *) malloc(C->size1*C->size2*sizeof(double));
//  gsl_matrix_to_cula(cA,A);
//  gsl_matrix_to_cula(cB,B);
//  gsl_matrix_to_cula(cC,C);
  //call mult
#ifdef CULA
  if ((A->m*A->n) > THRESH)
    {
      //printf("Moving to GPU %d\n",A->m*A->n);
      s = culaDgemm(TA,TB,m,n,k,alpha,A->data,lda,B->data,ldb,beta,C->data,ldc);
      if(s != culaNoError)
	{
	  printf("%s\n", culaGetStatusString(s));
	  fprintf(stdout,"****Error in Multiplication %d\n",culaGetErrorInfo(s));
	  /* ... Error Handling ... */
	}
    }
  else
#endif
  //#else
  //  dgemm_(&TA,&TB,&m,&n,&k,&alpha,A->data,&lda,B->data,&ldb,&beta,C->data,&ldc);
    dgemm_(&TA,&TB,&m,&n,&k,&alpha,A->data,&lda,B->data,&ldb,&beta,C->data,&ldc);
  //#endif
  //  cula_to_gsl_matrix(C,cC);
  return 0;
}

void my_symcheck(struct my_matrix * mat)
{
  int i,j;
  double assym;
  for (i = 0; i < mat->m; i++)
    for (j = 0; j < i; j++)
      {
	assym = fabs((mat->data[i*mat->m + j] - mat->data[j*mat->m + i])/(mat->data[i*mat->m + j] + mat->data[j*mat->m + i]));
	if (assym > 0.00001)
	  printf("ASSYM %f\n",assym);
      }

}

int get_inverse_lu(struct my_matrix *m, struct my_matrix *lu, int dim, double *det)
{
  char uplo;
  int N,i,j,lda;
  //  double *A;
  int * ipiv;
  ipiv = (int *) malloc(dim*sizeof(int));
#ifdef CULA
  culaStatus s;
#else
  int info;
#endif
  uplo = 'L';
  N = dim;
  lda = N;
#ifndef CULA
  double * work;
  int lwork = m->m*5;
  work = (double *) malloc(lwork*sizeof(double));
  dgetrf_(&N,&dim,m->data,&lda,ipiv,&info);
  
  if (info != 0)
    fprintf(stderr,"****Error in LU lapack %d \n",info);
  my_matrix_memcpy(lu,m);
  dgetri_(&N,m->data,&lda,ipiv,work,&lwork,&info);
  if (info != 0)
    fprintf(stderr,"****Error in Inverse lapack %d\n",info);
  free(work);
#endif


#ifdef CULA
  s = culaDgetrf(N,N,m->data,lda,ipiv);
  if(s != culaNoError)
    {
      printf("%s\n", culaGetStatusString(s));
      fprintf(stdout,"****Error in LU %d\n",culaGetErrorInfo(s));
      /* ... Error Handling ... */
    }
  else
    {
      ;
    }
  memcpy(lu->data,m->data,m->m*m->n*sizeof(double));
#endif
  //compute determinant
  double d = 0.0;
  double sign = 1.0;
  for (i = 0; i < dim; i++)
    {
      d += log(fabs(lu->data[i*lu->m + i]));
//      if (ipiv[i] != i)
//	sign = -sign;
//      else
//	d -= log(lu->data[i*lu->m + i]);
    }
  *det = sign*d;
#ifdef CULA
  s = culaDgetri(N,m->data,lda,ipiv);
  if(s != culaNoError)
    {
      printf("%s\n", culaGetStatusString(s));
      fprintf(stderr,"****Error in LU Inversion \n");
    }
#endif
  free(ipiv);
  lu->ulo = uplo;
  m->ulo = uplo;
  return 0;
}

//wrapper to input gsl matrix objects and perform inversion with cula
//int get_inverse_cholesky(gsl_matrix *m, gsl_matrix *cholesky, int dim)
int get_inverse_cholesky(struct my_matrix *m, struct my_matrix *cholesky, int dim)
{
  char uplo;
  int N,i,j,lda;
  //  double *A;
  int info;
#ifdef CULA
  culaStatus s;
#endif
  uplo = 'L';
  N = m->m;
  lda = N;

#ifdef SYMCHECK
  my_symcheck(m,dim);
#endif

#ifndef CULA
  double * AP;
  AP = (double *) malloc(N*(N+1)/2*sizeof(double));
  for (i = 1; i < (N+1); i++)
    for (j = 1; j <= i; j++)
      {
	AP[i+(j-1)*(2*N-j)/2-1] = m->data[(j-1)*m->m + (i-1)];
      }  
  dpptrf_(&uplo,&N,AP,&info);
  if (info != 0)
    fprintf(stderr,"****Error in Cholesky %d\n",info);
#endif
  //  A = (double *) malloc(N*N*sizeof(double));
  //  for (i = 1; i < (N+1); i++)
//    for (j = 1; j <= i; j++)
//      {
//	A[(j-1)*N+i-1] = gsl_matrix_get(m,i-1,j-1);
//	  //AP[i+(j-1)*(2*N-j)/2-1] = gsl_matrix_get(m,i-1,j-1);
//	//printf("%d %d %e\n",i,j,gsl_matrix_get(m,i,j));
//      }
//  gsl_matrix_to_cula(A,m);
  //  dpptrf_(uplo,&N,AP,&info);
#ifdef CULA
  s =  culaDpotrf(uplo,N,m->data,lda);

  if(s != culaNoError)
    {
      printf("%s\n", culaGetStatusString(s));
      fprintf(stdout,"****Error in Cholesky %d\n",culaGetErrorInfo(s));
      /* ... Error Handling ... */
    }
  else
    {
      ;
      //printf("No error in cholesky\n");
    }

  memcpy(cholesky->data,m->data,m->m*m->n*sizeof(double));
  s = culaDpotri(uplo,N,m->data,lda);
  if(s != culaNoError)
    {
      printf("%s\n", culaGetStatusString(s));
      fprintf(stderr,"****Error in Inversion \n");
    }
#endif
  cholesky->ulo = uplo;
  m->ulo = uplo;
#ifndef CULA
  for (i = 1; i < (N+1); i++)
    for (j = 1; j <= i; j++)
      {
	m->data[(j-1)*N+i-1] = AP[i+(j-1)*(2*N-j)/2-1];
      }  
  my_matrix_memcpy(cholesky,m);
  dpotri_(&uplo,&N,m->data,&lda,&info);
  if (info != 0)
    fprintf(stderr,"****Error in Inversion \n");
  free(AP);
#endif
//  for (i = 0; i < N; i++)
//    for (j = 0; j <= i; j++)
//      {
//	gsl_matrix_set(m,i,j,A[j*N+i]);
//	gsl_matrix_set(m,j,i,A[j*N+i]);
//      }
  //  cula_to_gsl_matrix_lo(m,A);
  //  free(A);
  return info;
}

void test_inversion(struct my_matrix *a, struct my_matrix *ainv)
{
  struct my_matrix * prod;
  prod = my_matrix_alloc(a->m,a->m);
  my_dgemm(CblasNoTrans,CblasNoTrans,1.0,a,ainv,0.0,prod);
  my_matrix_print("INVERSION TEST \0",prod);
  my_matrix_free(prod);
}

void test_inversion_sym(struct my_matrix *a, struct my_matrix *ainv)
{
  struct my_matrix * prod;
  prod = my_matrix_alloc(a->m,a->m);
  my_dsymm(CblasLeft,1.0,a,ainv,0.0,prod);
  my_matrix_print("INVERSION TEST \0",prod);
  my_matrix_free(prod);
}

