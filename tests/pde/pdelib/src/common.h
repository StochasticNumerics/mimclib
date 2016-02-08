#ifndef _COMMON_
#define _COMMON_

int nrand(PetscRandom rctx, int N, double *G);
int urand(PetscRandom rctx, int N, double *U);
double urand1(PetscRandom rctx);


static inline double POW2(double x) {return x*x;}
static inline double POW3(double x) {return x*x*x;}
static inline double max(double x, double y) {return x>y?x:y;}
static inline double min(double x, double y) {return x<y?x:y;}
static inline double sign(double x) {return x>0?1:-1;}

void legendre_quadrature(int *n, const double **points, const double **weights);  // Over interval [-1,1] with pdf 1/4
void hermite_quadrature(int *n, const double **points, const double **weights);   // Over interval [-inf,inf] with pdf exp(-x^2)

void myPetscInit(int argc, char *argv[]);
void myPetscFinal();
#endif
