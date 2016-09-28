#include "petsc.h"
#include "common.h"
#include "assert.h"
//////// This solver is based on Leveque book
#define MAXD 10
#define M_SQRT_PI 1.77245385091

#define DBOUT(str) printf("<<>> " str "\n");

typedef void *SField;
typedef const void *CSField;

#pragma GCC visibility push(default)
extern "C"{
void SFieldCreate(SField *_sf,
                         int problem_arg,
                         uint d,
                         double a0,
                         double f0,
                         double df_nu,
                         double df_L,
                         double df_sig,
                         double qoi_scale,
                         double *qoi_x0,
                         double qoi_sigma);
int SFieldBeginRuns(SField sfv,
                    unsigned int N,
                    const unsigned int *nelem);
double SFieldSolveFor(SField sfv, double *Y, unsigned int yCount);
void SFieldEndRuns(SField sfv);
void SFieldDestroy(SField *_sf);
}
#pragma GCC visibility pop

struct __sfield {
    // Problem parameters
    double a0, f0, df_nu, df_L, df_sig;
    unsigned int d, problem_arg;

    // QoI
    double sigma2, x0[MAXD], qoi_scale;

    /// Runs
    unsigned int maxN;
    int curN;
    double *Y;
    ind_t *N_multi_idx;
    unsigned int modes;
    ind_t *d_multi_idx;


    Mat J;
    Vec U,F;
    KSP ksp;

    // Our mesh is equi-spaced with mesh[d] points
    // constant mesh size with first element at x_0=0 and last at x_{m+1}=1
    // and m intervals
    int mesh[MAXD];
    PetscReal timeAssembly;
    PetscReal timeSolver;

    int running;
};

typedef struct __sfield *mySField;
typedef const struct __sfield *myCSField;

void SFieldCreate(SField *_sf,
                  int problem_arg,
                  uint d,
                  double a0,
                  double f0,
                  double df_nu,
                  double df_L,
                  double df_sig,
                  double qoi_scale,
                  double *qoi_x0,
                  double qoi_sigma){
    mySField sf = new __sfield;
    *_sf = sf;

    sf->problem_arg = problem_arg;
    assert(sf->problem_arg == 0 || sf->problem_arg == 1);

    sf->d = d;
    sf->a0 = a0;
    sf->f0 = f0;
    sf->df_sig = df_sig;
    sf->df_nu = df_nu;
    sf->df_L = df_L;
    unsigned int i;
    for (i=0;i<sf->d;i++)
        sf->x0[i] = qoi_x0[i];

    sf->qoi_scale = qoi_scale;
    sf->sigma2 = qoi_sigma<0?-1:qoi_sigma*qoi_sigma;
    sf->running = 0;

    unsigned int total = 1 << d;
    sf->d_multi_idx = new ind_t[total * d];
    ind_t m[sf->d];
    for (i=0;i<sf->d;i++) m[i]=1;
    TensorGrid(d, 0, m, sf->d_multi_idx, total);
}

int linIdx_Sys(int d, const int* mesh, int* pt, int i, int m){
    int j, idx = 0, c=1;
    for (j=0;j<d;j++) {
        idx += ((pt[j]-1) + (i==j)*m) * c;
        c *= mesh[j];
    }
    return idx;
}

// -------------------------------------------
PetscReal CalcCoeff(const PetscReal* x, myCSField sf){
    int curY = 0;   // Every time we increment we should check that is
                    // is less than N. We are ignoring the first mode
                    // because the coefficient is zero
    unsigned int i, j, jj;
    unsigned int per_mode = (1 << sf->d);
    double field=0;
    for (i=0;i<sf->modes && curY < sf->curN;i++){
        double cos_k[sf->d];
        double sin_k[sf->d];
        uint k0=0;
        uint k1=0;
        ind_t* k = sf->N_multi_idx+i*sf->d;
        for (j=0;j<sf->d;j++){
            cos_k[j] = cos(k[j]*x[j]*M_PI/sf->df_L);
            sin_k[j] = sin(k[j]*x[j]*M_PI/sf->df_L);
            k0 += k[j]>0;
            k1 += k[j];
        }
        double Ak = sf->df_sig * pow(2, k0/2.) * pow(1 + k1*k1, -(sf->df_nu+sf->d/2.)/2.);
        for (jj=0;jj<per_mode && curY < sf->curN;jj++){
            ind_t* l = sf->d_multi_idx + jj*sf->d;
            double temp = 1;

            bool skip_Y = 0;
            for (j=0;j<sf->d;j++){
                skip_Y = skip_Y || (l[j]==0 && k[j]==0)
                    || (sf->problem_arg == 1 && l[j] == 0);
                temp *= pow(cos_k[j], l[j])*pow(sin_k[j], 1-l[j]);
            }
            if (skip_Y){
                // Do not increment curY
                continue;
            }
            //printf("curY=%d -> k=%d, Ak=%.12f\n", curY, i, Ak);
            /* printf("%d -> %d -> Ak=%.12g, Y=%.12g, temp=%.12g, cos=%.12g, sin=%.12g, l=%d -> %.12g\n", k[0], */
            /*        curY, Ak, sf->Y[curY], temp, cos_k[0], sin_k[0], l[0], Ak*sf->Y[curY]*temp); */
            field += Ak*sf->Y[curY++]*temp;
        }
    }
    assert(curY == sf->curN || curY+1 == sf->curN);
    return sf->a0 + exp(field);
}

PetscReal QoIAtPoint(const PetscReal *x, PetscReal u, myCSField sf) {
    double c = sf->d * sf->qoi_scale;
    double sigma2 = sf->sigma2;
    if (sigma2 < 0)
        return c*u;
    const PetscReal *x0 = sf->x0;
    //printf("x0=%f, sigma2=%f, sf->d=%d\n", sf->x0[0], sigma2, sf->d);
    unsigned int i=0;
    for (i=0;i<sf->d;i++)
        c *= exp(-0.5*(((x[i]-x0[i])*(x[i]-x0[i]))/sigma2));
    return c*u*pow(2*sigma2*M_PI, -static_cast<float>(sf->d)/2.);
}

// ------------------------------------------
static inline double dx(int i, myCSField sf) {
    return 1./(double)(sf->mesh[i]+1);
}
void getPoint(const int *pt, double *x, myCSField sf) {
    unsigned int i;
    assert(sf->d < MAXD);
    for (i=0;i<sf->d;i++){
        x[i] = (double)pt[i] * dx(i, sf);
    }
}

double Coeff(int *pt, int di, double shift, myCSField sf) {
    double x[sf->d];
    getPoint(pt, x, sf);
    x[di] += shift;
    /* double xx=0.5; */
    /* printf("x=%.12f, field=%f\n", xx, CalcCoeff(&xx, sf)); */
    /* assert(0); */
    return CalcCoeff(x, sf);
}

double Forcing(int *pt, myCSField sf) {
    if (sf->problem_arg == 0)
        return sf->f0;
    double x[sf->d];
    getPoint(pt, x, sf);
    double c = 1.;
    unsigned int i=0;
    double sigma2 = 0.1*0.1;
    for (i=0;i<sf->d;i++)
        c *= exp(-0.5*(((x[i] - 0.5)*(x[i] - 0.5))/sigma2));
    return sf->f0 * c * pow(2*sigma2*M_PI, -sf->d/2.);
}

double Integrate(Vec U, int *pt, unsigned int i, myCSField sf) {
    if (i < sf->d){
        double g = 0;
        // Trapezoidal rule: \int_0^1 f(x) = (h/2) * \sum_{i=0}^{m} (f(x_i)+f(x_{i+1}))
        // In the special case where f(0) = f(1) = f(x_0) = f(x_{m+1}) = 0
        // \int_0^1 f(x) = 0.5*f(x_1) + {  \sum_{i=1}^{m-1} 0.5*(f(x_i)+f(x_{i+1}))  } + 0.5*f(x_m)
        double h_2 = (dx(i, sf)/2.);

        // u(x_0) = 0
        pt[i] = 1;
        g += Integrate(U, pt, i+1, sf) * h_2;
        for (pt[i]=1;pt[i]<sf->mesh[i];pt[i]++) {
            g += Integrate(U, pt, i+1, sf) * h_2;
            pt[i] += 1;
            g += Integrate(U, pt, i+1, sf) * h_2;
            pt[i] -= 1;
        }
        pt[i] = sf->mesh[i];
        g += Integrate(U, pt, i+1, sf) * h_2;
        // u(x_{m+1}) = 0
        return g;
    }
    int r = linIdx_Sys(sf->d, sf->mesh, pt,0,0);
    double x[sf->d];
    double Ux;
    getPoint(pt, x, sf);
    VecGetValues(U,1,&r,&Ux);
    return QoIAtPoint(x, Ux, sf);
}


int SFieldBeginRuns(SField sfv,
                    unsigned int N,
                    const unsigned int *nelem) {
    mySField sf = static_cast<mySField>(sfv);
    unsigned int j;
    assert(!sf->running);
    sf->maxN = N;
    sf->running = 1;
    sf->timeAssembly = 0;
    sf->timeSolver = 0;

    // Given k indices we need k*2^d variables
    // Given N variables we need N / 2^d
    // Now we are overestimating the number of modes.
    sf->modes = sf->maxN;//(sf->N+ignored_modes) / (1 << sf->d) + ((sf->N+ignored_modes) % (1 << sf->d) != 0);
    sf->N_multi_idx = new ind_t[sf->modes * sf->d];
    GenTDSet(sf->d, 0, sf->N_multi_idx, sf->modes);

    // Create sparse Matrix of size prod(mesh)
    Mat J; Vec F; Vec U;
    int s=1;
    for (j=0;j < sf->d;j++){
        sf->mesh[j] = nelem[j];
        s *= sf->mesh[j];
    }
    MatCreate(PETSC_COMM_WORLD,&J);
    MatSetSizes(J,s,s,s,s);
    MatSetType(J,MATSEQAIJ);
    MatSeqAIJSetPreallocation(J,1+2*sf->d,NULL);
    MatSetFromOptions(J);

    /* MatSetType(J,MATSEQDENSE); */
    /* MatSeqDenseSetPreallocation(J,NULL); */
    /* MatSetFromOptions(J); */
    MatSetUp(J);
    VecCreate(PETSC_COMM_WORLD,&F);
    VecSetSizes(F,PETSC_DECIDE,s);
    VecSetFromOptions(F);
    VecSetUp(F);
    VecDuplicate(F,&U);
    KSP ksp;
    KSPCreate(PETSC_COMM_WORLD,&ksp);
    KSPSetFromOptions(ksp);

    sf->J = J; sf->F = F; sf->U = U; sf->ksp = ksp;
    return 0;
}

void JacobianOnD(Mat J, Vec F, unsigned int i, int* pt, myCSField sf){
    if (i < sf->d){
        for (pt[i]=1;pt[i]<=sf->mesh[i];pt[i]++)
            JacobianOnD(J, F, i+1, pt, sf);
        return;
    }
    int r = linIdx_Sys(sf->d, sf->mesh, pt,0,0);
    unsigned int j;
    // a' u' + a u'' = F
    for (j=0;j<sf->d;j++){
        double ldx = dx(j, sf);
        double dx2 = ldx*ldx;
        double ap = Coeff(pt,j,+ldx/2., sf);
        double am = Coeff(pt,j,-ldx/2., sf);

        // Our operators are isotropic
        if (pt[j] > 1){
            int c = linIdx_Sys(sf->d, sf->mesh, pt, j, -1);
            MatSetValue(J, r, c, am/dx2, ADD_VALUES);
        }
        MatSetValue(J, r, r, -(am+ap)/dx2, ADD_VALUES);
        if (pt[j] < sf->mesh[j]){
            int c = linIdx_Sys(sf->d, sf->mesh, pt, j, 1);
            MatSetValue(J, r, c, ap/dx2, ADD_VALUES);
        }
    }

    double forcing = Forcing(pt, sf);
    VecSetValue(F, r, -forcing, INSERT_VALUES);
}

double SFieldSolveFor(SField sfv, double *Y, unsigned int yCount) {
    mySField sf = static_cast<mySField>(sfv);
    assert(yCount <= sf->maxN);
    assert(Y);
    assert(sf->running);
    sf->Y = Y;
    sf->curN = yCount;

    // -------------- SOLVE
    PetscErrorCode ierr;
    PetscLogDouble tic,toc;
    PetscTime(&tic);
    int pt[sf->d];

    ierr = MatZeroEntries(sf->J); CHKERRQ(ierr);
    JacobianOnD(sf->J, sf->F, 0, pt, sf);

    ierr = MatAssemblyBegin(sf->J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(sf->J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    PetscTime(&toc);
    sf->timeAssembly += toc-tic;
    PetscTime(&tic);
    ierr = VecZeroEntries(sf->U);  CHKERRQ(ierr);
    ierr = KSPSetOperators(sf->ksp, sf->J, sf->J); CHKERRQ(ierr);
    ierr = KSPSetUp(sf->ksp);  CHKERRQ(ierr);
    ierr = KSPSolve(sf->ksp,sf->F,sf->U); CHKERRQ(ierr);
    PetscTime(&toc);
    sf->timeSolver  += toc-tic;
    return Integrate(sf->U,pt,0,sf);
}

void SFieldEndRuns(SField sfv) {
    mySField sf = static_cast<mySField>(sfv);
    assert(sf->running);
    PetscErrorCode ierr;
    ierr = KSPDestroy(&sf->ksp);CHKERRV(ierr);
    ierr = MatDestroy(&sf->J);CHKERRV(ierr);
    ierr = VecDestroy(&sf->F);CHKERRV(ierr);
    ierr = VecDestroy(&sf->U);CHKERRV(ierr);

    delete [] sf->N_multi_idx;
    sf->running = 0;
    sf->N_multi_idx = 0;
}

void SFieldDestroy(SField *_sf) {
    mySField sf = static_cast<mySField>(*_sf);
    *_sf = 0;
    if (sf->running){
        fprintf(stderr, "WARNING: must end runs before\n");
        SFieldEndRuns(sf);
    }
    delete[] sf->d_multi_idx;
    delete sf;
}
