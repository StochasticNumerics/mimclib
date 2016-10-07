#include "petsc.h"
#include "common.hpp"
#include "assert.h"

//////// this solver is based on Leveque book
#define MAXD 10
#define M_SQRT_PI 1.77245385091

#define DBOUT(str) printf("<<>> " str "\n");

typedef void *SField;
typedef const void *CSField;
typedef unsigned long long int uint64;

// These functions are taken from set_util.so
typedef unsigned short ind_t;
typedef unsigned int uint32;
void GenTDSet(ind_t d, ind_t base, ind_t *td_set, uint32 count);

#pragma GCC visibility push(default)
extern "C"{
PetscErrorCode SFieldCreate(SField *);
int SFieldBeginRuns(SField, const unsigned int*, unsigned int);
int SFieldSolveFor(SField, double *, unsigned int, double *, int);
int SFieldEndRuns(SField);
int SFieldGetN(SField sfv);
int SFieldGetDim(SField sfv);
int SFieldDestroy(SField *);
}
#pragma GCC visibility pop

struct RunData {
    Mat J;
    Vec U,F;
    KSP ksp;

    // Our mesh is equi-spaced with mesh[d] points
    // constant mesh size with first element at x_0=0 and last at x_{m+1}=1
    // and m intervals
    int mesh[MAXD];
    PetscReal timeAssembly;
    PetscReal timeSolver;

    SField sf;
    double modifier;
};

typedef struct RunData *RunContext;
typedef const struct RunData *CRunContext;

struct __sfield {
    int d;
    int N;

    double a0;    // Coefficients
    double f0;
    double sigma2, x0[MAXD];
    double qoi_scale;
    double L;

    /// Runs
    RunContext run_data;
    int running;

    ind_t *multi_idx;
    double *Y;
    double df_L, df_rho, df_p1, df_p2, df_sig;
    int df_shift;
};

typedef struct __sfield *mySField;
typedef const struct __sfield *myCSField;

PetscErrorCode SFieldCreate(SField *_sf){
    PetscErrorCode ierr;
    mySField sf = new  __sfield;
    *_sf = sf;

    sf->a0 = 0.01;
    sf->f0 = 1;
    sf->qoi_scale = 1;
    sf->x0[0] = M_PI/5.;
    sf->x0[1] = 1./sqrt(3);
    sf->x0[2] = sqrt(2.)/3.;
    sf->d = 3;
    sf->N = 1;

    sf->df_p1 = -2;
    sf->df_p2 = -1;
    sf->df_rho = 0.2;
    sf->df_sig = 1.;
    sf->df_L = 1.;
    sf->df_shift = 0;
    int x0_dim=MAXD;
    double sigma = 0.01;
    PetscBool x0_set;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"qoi_","Options for program","Program");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-dim","Number of dimensions",__FILE__,sf->d,&sf->d,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-sigma","Sigma of QoI",__FILE__,sigma,&sigma,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-scale","Final scaling of QoI",__FILE__,sf->qoi_scale,&sf->qoi_scale,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsRealArray("-x0","x0 of QoI",__FILE__,sf->x0,&x0_dim,&x0_set);CHKERRQ(ierr);
    ierr  = PetscOptionsReal("-a0","",__FILE__,sf->a0,&sf->a0,PETSC_NULL);CHKERRQ(ierr);
    ierr  = PetscOptionsReal("-f0","",__FILE__,sf->f0,&sf->f0,PETSC_NULL);CHKERRQ(ierr);
    ierr  = PetscOptionsInt("-N","",__FILE__,sf->N,&sf->N,PETSC_NULL);CHKERRQ(ierr);
    ierr  = PetscOptionsInt("-df_shift","",__FILE__,sf->df_shift,&sf->df_shift,PETSC_NULL);CHKERRQ(ierr);
    ierr  = PetscOptionsReal("-df_rho","",__FILE__,sf->df_rho,&sf->df_rho,PETSC_NULL);CHKERRQ(ierr);
    ierr  = PetscOptionsReal("-df_p1","",__FILE__,sf->df_p1,&sf->df_p1,PETSC_NULL);CHKERRQ(ierr);
    ierr  = PetscOptionsReal("-df_p2","",__FILE__,sf->df_p2,&sf->df_p2,PETSC_NULL);CHKERRQ(ierr);
    ierr  = PetscOptionsReal("-df_sig","",__FILE__,sf->df_sig,&sf->df_sig,PETSC_NULL);CHKERRQ(ierr);
    ierr  = PetscOptionsReal("-df_L","",__FILE__,sf->df_L,&sf->df_L,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);

    sf->sigma2 = sign(sigma)*sigma*sigma;
    assert(sf->d < MAXD);
    assert(!x0_set || x0_dim >= sf->d);

    sf->running = 0;

    sf->multi_idx = new ind_t[sf->d*sf->N];
    GenTDSet(sf->d, 1, sf->multi_idx, sf->N);
    return 0;
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
PetscReal lambda(PetscInt k, myCSField ctx){
    /* double w; */
    /* int kk = k/2 + 1; */
    /* if (k == 0) */
    /*     w =  ctx->df_sig * sqrt(0.5 *ctx->df_rho * M_SQRT_PI / ctx->df_L); */
    /* else */
    /*     w = ctx->df_sig * sqrt((ctx->df_rho * M_SQRT_PI / ctx->df_L) * */
    /*                            exp(-POW2((kk)*M_PI*ctx->df_rho/(2*ctx->df_L)))); */
    /* return w; */
    return exp(-k);
}

PetscReal phi(PetscInt k, PetscReal x, myCSField ctx){
    if (k == 0)
        return 1.;
    double arg = (k/2)*M_PI/ctx->df_L;
    if (k%2 == 1) return cos(x*arg);
    else          return sin(x*arg);
}

PetscReal phi_x(PetscInt k, PetscReal x, myCSField ctx){
    if (k == 0)
        return 0;
    double arg = (k/2)*M_PI/ctx->df_L;
    if (k%2 == 1)   return -arg * sin(x*arg);
    else            return arg * cos(x*arg);
}

PetscReal Solution(int d_i, int dd, PetscReal* x, myCSField ctx){
    // d_i is the dimension to take derivative w.r.t.
    // dd is the number of times to take the derivatives.
    // dd = 1 or 2
    assert(dd >= 0 && dd <= 2);
    double arg1=1, arg2=1, arg3=0, arg4=0, arg5=0;
    int i;
    for (i=0;i<ctx->d;i++){
        if (i == d_i){
            if (dd == 1){
                arg1 *= POW2(x[i]) * (4*x[i]-3);;
                arg2 *= 2*x[i]*M_PI*cos(M_PI * POW2(x[i]));
                arg5 += 2*x[i]-1;
                continue;
            }
            if (dd == 2){
                arg1 *= 6 * x[i] * (2*x[i]-1);
                arg2 *= 2*M_PI*(cos(M_PI * POW2(x[i])) -
                                2*M_PI*POW2(x[i])*sin(M_PI * POW2(x[i])));
                arg5 += 2;
                continue;
            }
        }
        arg1 *= POW3(x[i])*(x[i]-1);
        arg2 *= sin(M_PI * POW2(x[i]));
        arg5 += x[i]*(x[i]-1);
    }
    int j;
    for (j=0;j<ctx->N;j++){
        arg3 += pow(j+ctx->df_shift+1, ctx->df_p1) * ctx->Y[j];
        arg4 += pow(j+ctx->df_shift+1, ctx->df_p2) * ctx->Y[j];
    }
    if (ctx->df_shift)
        return arg1 * exp(arg3);    // A hack for testing
    return arg1 * exp(arg3) +  arg2 * exp(arg4);
    //return arg5;
}

PetscReal EvaluateField(int derivative, const PetscReal* x,
                        myCSField ctx){
    int i,j;
    double field = 0;
    for (j=0;j<ctx->N;j++){
        double Y = ctx->Y[j];
        double c_lambda = lambda(j, ctx);
        double prod_phi=1;
        for (i=0;i<ctx->d;i++){
            if (i == derivative)
                prod_phi *= phi_x(ctx->multi_idx[j*ctx->d+i], x[i], ctx);
            else
                prod_phi *= phi(ctx->multi_idx[j*ctx->d+i], x[i], ctx);
        }
        field += c_lambda * prod_phi * Y;
    }
    return field;
}

PetscReal CalcCoeff(int derivative, const PetscReal* x, myCSField ctx){
    double field = EvaluateField(-1, x, ctx);
    if (derivative < 0)
        return ctx->a0 + exp(field);
    return EvaluateField(derivative, x, ctx) * exp(field);
}

PetscReal QoIAtPoint(const PetscReal *x, PetscReal u, myCSField ctx) {
    double c=ctx->qoi_scale;
    double sigma2 = ctx->sigma2;
    if (sigma2 < 0)
        return c*u;
    const PetscReal *x0 = ctx->x0;
    int i=0;
    for (i=0;i<ctx->d;i++)
        c *= exp(-0.5*(POW2(x[i]-x0[i])/sigma2));
    return c*u*pow(2*sigma2*M_PI, -ctx->d/2.);
}

// ------------------------------------------
static inline double dx(int i, CRunContext ctx) {
    return 1./(double)(ctx->mesh[i]+1);
}
void getPoint(const int *pt, double *x, CRunContext ctx) {
    int i;
    assert(((myCSField)ctx->sf)->d < MAXD);
    for (i=0;i<((myCSField)ctx->sf)->d;i++){
        x[i] = (double)pt[i] * dx(i, ctx);
    }
}

double Coeff(int *pt, int di, double shift, CRunContext ctx) {
    double x[((myCSField)ctx->sf)->d];
    getPoint(pt, x, ctx);
    x[di] += shift;
    return CalcCoeff(-1, x, ((myCSField)ctx->sf));
}

double Forcing(int *pt, CRunContext ctx) {
    myCSField sf = (myCSField)ctx->sf;
    double x[sf->d];
    getPoint(pt, x, ctx);

    double F = 0;
    double a = CalcCoeff(-1, x, sf);
    int i;
    for (i=0;i<sf->d;i++){
        F += -a*Solution(i, 2, x, sf) - CalcCoeff(i, x, sf) * Solution(i,1,x, sf);
    }
    return F;
}

double Integrate(Vec U, int *pt, int i, CRunContext ctx) {
    myCSField sf = ((myCSField)ctx->sf);
    if (i < sf->d){
        double g = 0;
        // Trapezoidal rule: \int_0^1 f(x) = (h/2) * \sum_{i=0}^{m} (f(x_i)+f(x_{i+1}))
        // In the special case where f(0) = f(1) = f(x_0) = f(x_{m+1}) = 0
        // \int_0^1 f(x) = 0.5*f(x_1) + {  \sum_{i=1}^{m-1} 0.5*(f(x_i)+f(x_{i+1}))  } + 0.5*f(x_m)
        double h_2 = (dx(i, ctx)/2.);

        // u(x_0) = 0
        pt[i] = 1;
        g += Integrate(U, pt, i+1, ctx) * h_2;
        for (pt[i]=1;pt[i]<ctx->mesh[i];pt[i]++) {
            g += Integrate(U, pt, i+1, ctx) * h_2;
            pt[i] += 1;
            g += Integrate(U, pt, i+1, ctx) * h_2;
            pt[i] -= 1;
        }
        pt[i] = ctx->mesh[i];
        g += Integrate(U, pt, i+1, ctx) * h_2;
        // u(x_{m+1}) = 0
        return g;
    }
    int r = linIdx_Sys(sf->d, ctx->mesh, pt,0,0);
    double x[sf->d];
    double Ux;
    getPoint(pt, x, ctx);
    VecGetValues(U,1,&r,&Ux);
    return QoIAtPoint(x, Ux, sf);
}


int SFieldBeginRuns(SField sfv, const unsigned int *nelem, unsigned int count) {
    mySField sf = static_cast<mySField>(sfv);
    int i,j;
    assert(!sf->running);

    sf->running = count;
    sf->run_data = new RunData[count];
    for (i=0;i<sf->running;i++) {
        RunContext r = &sf->run_data[i];
        // Hook up the stochastic field to the context
        r->timeAssembly = 0;
        r->timeSolver = 0;
        r->sf  = sf;

        // Create sparse Matrix of size prod(mesh)
        Mat J; Vec F; Vec U;

        int s=1;
        for (j=0;j < sf->d;j++){
            r->mesh[j] = nelem[i*sf->d+j];
            s *= r->mesh[j];
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

        r->J = J; r->F = F; r->U = U; r->ksp = ksp;
    }
    return 0;
}

void JacobianOnD(Mat J, Vec F, int i, int* pt, CRunContext ctx){
    myCSField sf = (myCSField) ctx->sf;
    if (i < sf->d){
        for (pt[i]=1;pt[i]<=ctx->mesh[i];pt[i]++)
            JacobianOnD(J, F, i+1, pt, ctx);
        return;
    }
    int r = linIdx_Sys(sf->d, ctx->mesh, pt,0,0);
    int j;
    // a' u' + a u'' = F
    for (j=0;j<sf->d;j++){
        double ldx = dx(j, ctx);
        double dx2 = ldx*ldx;
        double ap = Coeff(pt,j,+ldx/2., ctx);
        double am = Coeff(pt,j,-ldx/2., ctx);

        // Our operators are isotropic
        if (pt[j] > 1){
            int c = linIdx_Sys(sf->d, ctx->mesh, pt, j, -1);
            MatSetValue(J, r, c, am/dx2, ADD_VALUES);
        }
        MatSetValue(J, r, r, -(am+ap)/dx2, ADD_VALUES);
        if (pt[j] < ctx->mesh[j]){
            int c = linIdx_Sys(sf->d, ctx->mesh, pt, j, 1);
            MatSetValue(J, r, c, ap/dx2, ADD_VALUES);
        }
    }

    double forcing = 1.;//Forcing(pt, ctx);
    VecSetValue(F, r, -forcing, INSERT_VALUES);
}


/* int SFieldGetSol(SField sfv, */
/*                  double *Y, unsigned int yCount, */
/*                  //double *diff_field, unsigned int df_mesh, */
/*                  double *sol, int sol_size, */
/*                  double *x_grd){ */
/*     mySField sf = (mySField)sfv; */
/*     assert(yCount == sf->N); */
/*     assert(Y && sol); */
/*     assert(sf->running == 1); */
/*     assert(sf->d == 1); */

/*     sf->Y = Y; */
/*     /\* sf->diff_field = diff_field; *\/ */
/*     /\* sf->df_mesh = df_mesh; *\/ */

/*     //CheckPhi(sf); */
/*     /\* assert(sf->N <= sf->df_mesh); *\/ */

/*     PetscErrorCode ierr; */

/*     // -------------- SOLVE */
/*     int i=0; */
/*     RunContext r = &sf->run_data[i]; */
/*     PetscLogDouble tic,toc; */
/*     PetscTime(&tic); */
/*     int pt[sf->d]; */

/*     ierr = MatZeroEntries(r->J); CHKERRQ(ierr); */
/*     JacobianOnD(r->J, r->F, 0, pt, r); */

/*     ierr = MatAssemblyBegin(r->J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr); */
/*     ierr = MatAssemblyEnd(r->J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr); */
/*     //MatSetOption(r->J,MAT_NEW_NONZERO_LOCATIONS,PETSC_TRUE); */
/*     //MatView(r->J, PETSC_VIEWER_STDOUT_SELF); */

/*     PetscTime(&toc); */
/*     r->timeAssembly += toc-tic; */
/*     PetscTime(&tic); */
/*     ierr = VecZeroEntries(r->U);  CHKERRQ(ierr); */
/*     ierr = KSPSetOperators(r->ksp, r->J, r->J); CHKERRQ(ierr); */
/*     ierr = KSPSetUp(r->ksp);  CHKERRQ(ierr); */
/*     ierr = KSPSolve(r->ksp,r->F,r->U); CHKERRQ(ierr); */
/*     PetscTime(&toc); */
/*     r->timeSolver  += toc-tic; */

/*     KSPConvergedReason reason=-1; */
/*     ierr = KSPGetConvergedReason(r->ksp, &reason); CHKERRQ(ierr); */
/*     assert(reason > 0); */

/*     PetscScalar *x; */
/*     int j; */
/*     PetscInt size; */
/*     VecGetSize(r->U, &size); */
/*     assert(sol_size == size); */
/*     VecGetArray(r->U, &x); */
/*     for (j=0;j<size;j++){ */
/*         sol[j] = x[j]; */
/*         pt[0] = j; */
/*         getPoint(pt, &x_grd[j], r); */
/*     } */
/*     VecRestoreArray(r->U, &x); */
/*     return 0; */
/* } */

int SFieldSolveFor(SField sfv, double *Y,
                   unsigned int yCount,
                   double *goals,
                   int goals_size) {
    mySField sf = static_cast<mySField>(sfv);
    assert(yCount == static_cast<unsigned int>(sf->N));
    assert(goals_size >= sf->running);
    assert(Y && goals);

    sf->Y = Y;

    //CheckPhi(sf);
    //assert(sf->N <= sf->df_mesh);
    PetscErrorCode ierr;
    // -------------- SOLVE
    int i;
    for (i=0;i<sf->running;i++)
    {
        RunContext r = &sf->run_data[i];
        PetscLogDouble tic,toc;
        PetscTime(&tic);
        int pt[sf->d];

        ierr = MatZeroEntries(r->J); CHKERRQ(ierr);
        JacobianOnD(r->J, r->F, 0, pt, r);

        ierr = MatAssemblyBegin(r->J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(r->J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        //MatSetOption(r->J,MAT_NEW_NONZERO_LOCATIONS,PETSC_TRUE);
        //MatView(r->J, PETSC_VIEWER_STDOUT_SELF);

        PetscTime(&toc);
        r->timeAssembly += toc-tic;
        PetscTime(&tic);
        ierr = VecZeroEntries(r->U);  CHKERRQ(ierr);
        ierr = KSPSetOperators(r->ksp, r->J, r->J); CHKERRQ(ierr);
        ierr = KSPSetUp(r->ksp);  CHKERRQ(ierr);
        //DEBOUT("KSPSolve");
        ierr = KSPSolve(r->ksp,r->F,r->U); CHKERRQ(ierr);
        //DEBOUT("Done!!");
        PetscTime(&toc);
        r->timeSolver  += toc-tic;

        *goals = Integrate(r->U,pt,0,r);
        goals++;
    }
    return 0;
}

int SFieldEndRuns(SField sfv) {
    mySField sf = static_cast<mySField>(sfv);
    assert(sf->running);
    PetscErrorCode ierr;
    int i;
    for (i=0;i<sf->running;i++)
    {
        RunContext r = &sf->run_data[i];
        //printf("%d, %.12e,\t%.12e,\n", r->mesh[0], r->timeAssembly, r->timeSolver);
        // Cleanup
        ierr = KSPDestroy(&r->ksp);CHKERRQ(ierr);
        ierr = MatDestroy(&r->J);CHKERRQ(ierr);
        ierr = VecDestroy(&r->F);CHKERRQ(ierr);
        ierr = VecDestroy(&r->U);CHKERRQ(ierr);
    }
    delete [] sf->run_data;
    sf->run_data = 0;
    sf->running = 0;
    return 0;
}

int SFieldDestroy(SField *_sf) {
    mySField sf = static_cast<mySField>(*_sf);
    *_sf = 0;
    if (sf->running){
        fprintf(stderr, "WARNING: must end runs before\n");
        SFieldEndRuns(sf);
    }

    delete [] sf->multi_idx;
    delete sf;
    return 0;
}

int SFieldGetDim(SField sfv) {
    mySField sf = static_cast<mySField>(sfv);
    return sf->d;
}

int SFieldGetN(SField sfv) {
    mySField sf = static_cast<mySField>(sfv);
    return sf->N;
}
