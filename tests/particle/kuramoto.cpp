#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <numeric>
#include <stdexcept>

typedef std::vector<double> array;
typedef unsigned int uint;

extern "C"
{
    void MultiKuramoto_QoI(unsigned int dim,
                  bool var_sig,
                    bool antithetic,
                    const uint* P_begin,
                    const uint* N_begin,
                    uint count,
                    double T, double K, double sig,
                    const double* initial_begin,
                    const double* disorder_begin,
                    const double* wiener_begin,
                    double *out_values);

    void* CreateRandGen(unsigned long long);
    void FreeRandGen(void*);
    void SampleKuramoto_QoI(void* pGen,
                        unsigned int dim,
                        bool var_sig,
                        bool antithetic,
                        const uint* P_begin,
                        const uint* N_begin,
                        uint count,
                        double T, double K, double sig,
                        uint M,
                        double *samples);

    void SampleKuramoto_Cov(void* pGen,
                            unsigned int dim,
                            bool var_sig,
                            bool antithetic,
                            const uint* P_begin,
                            const uint* N_begin,
                            uint count,
                            double T, double K, double sig,
                            uint M,
                            double* moments);
    void SolveFokkerPlanck1D(double sig, double K,
                             double xmax, uint Nx,
                             double T, uint Nt,
                             double *p0);
};

template<typename T>
class ndarray_C
{
public:
    ndarray_C(T* _data, size_t* _sizes, unsigned int d) {
        init(_data, _sizes, d);
    }
    ndarray_C(T* _data, unsigned int i, unsigned int j){
        size_t _sizes[] = {i,j};
        init(_data, _sizes, 2);
    }
    ndarray_C(T* _data, unsigned int i, unsigned int j, unsigned int k){
        size_t _sizes[] = {i,j,k};
        init(_data, _sizes, 3);
    }
    ndarray_C(T* _data, unsigned int i, unsigned int j, unsigned int k, unsigned int l){
        size_t _sizes[] = {i,j,k,l};
        init(_data, _sizes, 4);
    }

    void init(T* _data, size_t* _sizes, int d){
        data = _data;
        sizes = std::vector<size_t>(_sizes, _sizes+d);
    }

    T* data;
    std::vector<size_t> sizes;

    size_t linIdx(int* idx, unsigned int d) const{
        // Last index varies the fastest
        size_t linIdx = 0;
        if (d != sizes.size())
            throw std::runtime_error("Size mismatch");
        size_t cur=1;
        for (int i=d-1;i>=0;i--){
            linIdx += cur * idx[i];
            cur *= sizes[i];
        }
        return linIdx;
    }

    inline T operator() (int* idx, int d) const{
        return data[linIdx(idx, d)];
    }
    inline T& operator() (int* idx, int d){
        return data[linIdx(idx, d)];
    }
    inline T operator() (int i, int j){
        int idx[] = {i,j}; return this->operator()(idx, 2);
    }
    inline T operator() (int i, int j) const{
        int idx[] = {i,j}; return this->operator()(idx, 2);
    }
    inline T& operator() (int i, int j, int k){
        int idx[] = {i,j,k};
        return this->operator()(idx, 3);
    }
    inline T operator() (int i, int j, int k) const{
        int idx[] = {i,j,k};
        return this->operator()(idx, 3);
    }
    inline T& operator() (int i, int j, int k, int l) const{
        int idx[] = {i,j,k,l};
        return this->operator()(idx, 4);
    }
};

double pow2(double x){return x*x;}

std::vector<array> RunKuramoto(unsigned int dim,
                               bool var_sig, double T, double K, double sig,
                               const ndarray_C<const double> &initial,
                               const ndarray_C<const double> &disorder,
                               const ndarray_C<const double> &wiener,
                               unsigned int totalN, unsigned int N,
                               unsigned int P, unsigned int group){
    unsigned int perN = totalN/N;
    unsigned int firstP = group*P;
    std::vector<array> theta(dim);

    for (unsigned int d=0;d<dim;d++)
        theta[d] = array(P);
    for (unsigned int j=0;j<P;j++){
        for (unsigned int d=0;d<dim;d++){
            theta[d][j] = initial(j+firstP, d);
        }
    }

    double dt = T/N;

    // wiener: Particle_1 Particle_2 ... Particle_P
    // Particle_1: t_0, t_1, ... t_N
    // initial: P*dim
    // disorder: P*dim
    // wiener: P*N*dim
    for (unsigned int n=0;n<N;n++){
        std::vector<array> couple(dim);
        for (unsigned int d=0;d<dim;d++)
            couple[d] = array(P, 0);
        for (unsigned int j=0;j<P;j++){
            for (unsigned int k=0;k<P;k++){
                double distance=0;
                for (unsigned int d=0;d<dim;d++){
                    distance += pow2(fmod(theta[d][k]-theta[d][j], 2*M_PI));
                }
                distance = sqrt(distance);
                for (unsigned int d=0;d<dim;d++){
                    if (distance > 0)
                        couple[d][j] += sin(distance) *
                            fmod(theta[d][k]-theta[d][j], 2*M_PI)/distance;
                }
            }
        }
        for (unsigned int j=0;j<P;j++){
            array dW = array(dim, 0);
            for (unsigned int k=0;k<perN;k++)
                for (unsigned int d=0;d<dim;d++)
                    dW[d] += wiener(firstP+j, perN*n+k, d);

            for (unsigned int d=0;d<dim;d++){
                theta[d][j] += dt * (disorder(j+firstP, d) + K*couple[d][j]/P) +
                    sig * dW[d] * (var_sig?cos(theta[d][j]):1);
            }
        }
    }
    return theta;
}

double RunKuramoto_QoI(unsigned int dim,
                   bool var_sig, double T, double K, double sig,
                   const ndarray_C<const double> &initial,
                   const ndarray_C<const double> &disorder,
                   const ndarray_C<const double> &wiener,
                   unsigned int totalN, unsigned int N,
                   unsigned int P, unsigned int group){
    std::vector<array> theta = RunKuramoto(dim, var_sig, T, K, sig,initial,
                                           disorder, wiener, totalN,
                                           N, P, group);
    double qoi = 0.;
    for (unsigned int d=0;d<dim;d++)
    {
        double real=0;
        double imag=0;
        for (unsigned int j=0;j<P;j++){
            real += cos(theta[d][j]);
            imag += sin(theta[d][j]);
        }
        qoi += real/P;
        //qoi += pow2(real/P) + pow2(imag/P);
    }

    // Calculate covariance between the first two particles
    // float mean = 0.930960856618;
    // return (cos(theta[0][0])-mean)*cos(theta[0][1]-mean);

    return qoi/dim;
}

double MultiKuramoto_Covariance(unsigned int dim,
              bool var_sig,
              bool antithetic,
              const uint* P_begin,
              const uint* N_begin,
              uint count,
              double T, double K, double sig,
              const double* initial_begin,
              const double* disorder_begin,
              const double* wiener_begin)
{
    // d: dim, P: max(P), N: max(N)
    // initial_begin: P*dim
    // disorder_begin: P*dim
    // wiener_begin: P*N*dim
    // out_values: P

    unsigned int maxP = *std::max_element(P_begin, P_begin+count);
    unsigned int maxN = *std::max_element(N_begin, N_begin+count);

    ndarray_C<const double> initial(initial_begin, maxP, dim);
    ndarray_C<const double> disorder(disorder_begin, maxP, dim);
    ndarray_C<const double> wiener(wiener_begin, maxP, maxN, dim);

    bool valid=true;
    for (unsigned int i=0;valid && i<count;i++){
        valid = valid && (maxP % *(P_begin+i) == 0)
            && (maxN % *(N_begin+i) == 0);
        valid = valid && *(P_begin+i) >= 2;
    }
    if (!valid)
        throw std::runtime_error("Number of particles and time steps should multiple of the max");
    if (count != 2)
        throw std::runtime_error("count must be 2");
    if (*P_begin != *(P_begin+1)*2)
        throw std::runtime_error("Must start with the largest P");

    std::vector<std::vector<array> > thetas;
    for (unsigned int i=0;i<count;i++){
        unsigned int groups = antithetic?(maxP / *(P_begin+i)):1;
        for (unsigned int group=0;group<groups;group++)
            thetas.push_back(RunKuramoto(dim, var_sig, T, K, sig, initial,
                                         disorder, wiener, maxN,
                                         *(N_begin+i), *(P_begin+i), group));

    }
    // Check covariance between first and second particle
    // thetas[system][dim][particle]
    int i = 1;   // if i=2, then we have two different systems!
    return (cos(thetas[0][0][0]) - cos(thetas[1][0][0])) *
        (cos(thetas[0][0][1+(i-1)*maxP/2]) - cos(thetas[i][0][1]));
}


void MultiKuramoto_QoI(unsigned int dim,
              bool var_sig,
              bool antithetic,
              const uint* P_begin,
              const uint* N_begin,
              uint count,
              double T, double K, double sig,
              const double* initial_begin,
              const double* disorder_begin,
              const double* wiener_begin,
              double* out_values)
{
    // d: dim, P: max(P), N: max(N)
    // initial_begin: P*dim
    // disorder_begin: P*dim
    // wiener_begin: P*N*dim
    // out_values: P

    unsigned int maxP = *std::max_element(P_begin, P_begin+count);
    unsigned int maxN = *std::max_element(N_begin, N_begin+count);

    ndarray_C<const double> initial(initial_begin, maxP, dim);
    ndarray_C<const double> disorder(disorder_begin, maxP, dim);
    ndarray_C<const double> wiener(wiener_begin, maxP, maxN, dim);

    bool valid=true;
    for (unsigned int i=0;valid && i<count;i++)
        valid = valid && (maxP % *(P_begin+i) == 0)
            && (maxN % *(N_begin+i) == 0);
    if (!valid)
        throw std::runtime_error("Number of particles and time steps should multiple of the max");
    for (unsigned int i=0;i<count;i++){
        unsigned int groups = antithetic?(maxP / *(P_begin+i)):1;
        double total=0;
        for (unsigned int group=0;group<groups;group++)
            total += RunKuramoto_QoI(dim, var_sig, T, K, sig, initial,
                                 disorder, wiener, maxN,
                                 *(N_begin+i), *(P_begin+i), group);
        out_values[i] = (total/groups);
    }
}

#include <random>
#include <chrono>

void* CreateRandGen(unsigned long long seed){
    return new std::mt19937(seed);
}

void FreeRandGen(void* pGen){
    delete reinterpret_cast<std::mt19937*>(pGen);
}

void SampleKuramoto_QoI(void* pGen,
                    unsigned int dim,
                    bool var_sig,
                    bool antithetic,
                    const uint* P_begin,
                    const uint* N_begin,
                    uint count,
                    double T, double K, double sig,
                    uint M,
                    double *samples){
    std::mt19937 &gen = *reinterpret_cast<std::mt19937*>(pGen);
    unsigned int maxN = *std::max_element(N_begin, N_begin+count);
    unsigned int maxP = *std::max_element(P_begin, P_begin+count);

    std::normal_distribution<double> dist_wiener(0.0, sqrt(T/maxN));
    std::normal_distribution<double> dist_initial(0.0, 0.2);
    std::uniform_real_distribution<double> dist_disorder(-0.2, 0.2);

    std::vector<double> initial(maxP, 0);
    std::vector<double> disorder(maxP, 0);
    std::vector<double> wiener(maxN*maxP, 0);
    std::vector<double> out_values(count, 0);

    for (uint m=0;m<M;m++){
        for (uint i=0;i<maxP;i++)
            initial[i] = dist_initial(gen);
        for (uint i=0;i<maxP;i++)
            disorder[i] = dist_disorder(gen);
        for (uint i=0;i<maxN*maxP;i++)
            wiener[i] = dist_wiener(gen);

        MultiKuramoto_QoI(dim, var_sig, antithetic, P_begin, N_begin, count,
                 T, K, sig, &initial[0], &disorder[0], &wiener[0],
                 &samples[0]);

        samples += count;
    }
}


void SampleKuramoto_Cov(void* pGen,
                    unsigned int dim,
                    bool var_sig,
                    bool antithetic,
                    const uint* P_begin,
                    const uint* N_begin,
                    uint count,
                    double T, double K, double sig,
                    uint M,
                    double* moments){
    std::mt19937 &gen = *reinterpret_cast<std::mt19937*>(pGen);
    unsigned int maxN = *std::max_element(N_begin, N_begin+count);
    unsigned int maxP = *std::max_element(P_begin, P_begin+count);

    std::normal_distribution<double> dist_wiener(0.0, sqrt(T/maxN));
    std::normal_distribution<double> dist_initial(0.0, 0.2);
    std::uniform_real_distribution<double> dist_disorder(-0.2, 0.2);

    moments[0] = moments[1] = 0;

    std::vector<double> initial(maxP, 0);
    std::vector<double> disorder(maxP, 0);
    std::vector<double> wiener(maxN*maxP, 0);
    for (uint m=0;m<M;m++){
        for (uint i=0;i<maxP;i++)
            initial[i] = dist_initial(gen);
        for (uint i=0;i<maxP;i++)
            disorder[i] = dist_disorder(gen);
        for (uint i=0;i<maxN*maxP;i++)
            wiener[i] = dist_wiener(gen);

        double value = MultiKuramoto_Covariance(dim, var_sig, antithetic, P_begin, N_begin, count,
                 T, K, sig, &initial[0], &disorder[0], &wiener[0]);
        moments[0] += value;
        moments[1] += value*value;
    }
}


void SolveFokkerPlanck1D(double sig, double K,
               double xmax, uint Nx,
               double T, uint Nt,
               double *p0) {
    double dt = T/Nt;
    double dx = 2*xmax/(Nx-1);
    double dx2 = dx*dx;
    double sig2 = sig*sig;
    auto xi = [dx, xmax](uint i) {
        return i*dx - xmax;
    };

    std::vector<double> a(Nx, 0);  // diag_1
    std::vector<double> b(Nx, 0);  // diag0
    std::vector<double> c(Nx, 0);  // diag1

    std::vector<double> c_star(Nx, 0);
    std::vector<double> d_star(Nx, 0);
    std::vector<double> d(Nx);

    std::vector<double> A_s(Nx*Nx);
    std::vector<double> A_c(Nx*Nx);
    uint ii=0;
    for (uint i=0;i<Nx;i++){
        for (uint j=0;j<Nx;j++){
            A_s[ii] = K*sin(xi(j)-xi(i))*dx;
            A_c[ii] = -K*cos(xi(j)-xi(i))*dx;
            ii++;
        }
    }
    for (uint n=0;n<Nt;n++){
        ii=0;
        for (uint i=0;i<Nx;i++){
            double A=0, A_x=0;
            for (uint j=0;j<Nx;j++){
                A   += A_s[ii]*p0[j];
                A_x += A_c[ii]*p0[j];
                ii++;
            }
            // Time derivative
            b[i] += 1./dt;

            // Diffusion coeff
            a[i]  -= 0.5*sig2/dx2;
            b[i]  -= -0.5*2.*sig2/dx2;
            c[i]  -= 0.5*sig2/dx2;

            // Advection coeff
            b[i] += A_x;

            // // Advection coeff (upwind)
            //diag1[i]  += a/(2*dx);
            //diag_1[i] -= a/(2*dx);

            if (A>0){
                b[i]  += A/dx;
                a[i] -= A/dx;
            }
            else{
                b[i] -= A/dx;
                c[i] += A/dx;
            }
        }

        // Solve using https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
        for (uint i=0;i<Nx;i++)
            d[i] = p0[i]/dt;

        // This updates the coefficients in the first row
        // Note that we should be checking for division by zero here
        c_star[0] = c[0] / b[0];
        d_star[0] = d[0] / b[0];

        // Create the c_star and d_star coefficients in the forward sweep
        for (uint i=1; i<Nx; i++) {
            double m = 1.0 / (b[i] - a[i] * c_star[i-1]);
            c_star[i] = c[i] * m;
            d_star[i] = (d[i] - a[i] * d_star[i-1]) * m;
        }

        // This is the reverse sweep, used to update the solution vector f
        p0[Nx-1] = d_star[Nx-1];
        for (int i=Nx-2; i >= 0; i--)
            p0[i] = d_star[i] - c_star[i] * p0[i+1];

        std::fill(a.begin(), a.end(), 0);
        std::fill(b.begin(), b.end(), 0);
        std::fill(c.begin(), c.end(), 0);
    }
}
