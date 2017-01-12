#include <iostream>
#include <cmath>
#include <vector>
#include "var_list.hpp"

extern "C"{
    void sample_optimal_leg_pts(unsigned int N,
                                const VarSizeList* bases_indices,
                                double *X, double a, double b);
}
std::vector<double> legendre_pol(double X, unsigned int N, double a, double b){
    X = (X-(b+a)/2.) / ((b-a)/2.);
    std::vector<double> ret(N);
    unsigned int deg = N-1;
    ret[0] = 1.;
    if (N == 1)
        return ret;
    ret[1] = X;
    for (unsigned int n=1;n<deg;n++){
        ret[n+1] = 1. / (n + 1.) * ((2. * n + 1) * X * ret[n] - n * ret[n - 1]);
    }
    for (unsigned int n=0;n<deg;n++){
        ret[n+1] *= std::sqrt(2*(n+1) + 1);
    }
    return ret;
}

void sample_optimal_leg_pts(unsigned int N,
                            const VarSizeList* bases_indices,
                            double *X, double a, double b) {
    static std::mt19937 gen;
    assert(bases_indices->count() > 0);
    std::uniform_int_distribution<unsigned int> uni_int(0, bases_indices->count()-1);
    std::uniform_real_distribution<double> uni(0., 1.);

    unsigned int max_dim = bases_indices->max_dim();
    double acceptanceratio = 1./(4*std::exp(1));
    for (unsigned int i=0;i<N;i++){
        auto base_pol = bases_indices->get(uni_int(gen));
        for (unsigned int dim=0;dim<max_dim;dim++){
            bool accept = false;
            double Xreal = 0;
            while (!accept){
                double Xnext = (std::cos(M_PI * uni(gen)) + 1.) / 2.;
                double dens_prop_Xnext = 1. / (M_PI * std::sqrt(Xnext*(1 - Xnext)));
                Xreal = a + Xnext*(b-a);
                double dens_goal_Xnext = legendre_pol(Xreal, 1+base_pol[dim], a, b).back();
                dens_goal_Xnext *= dens_goal_Xnext;
                double alpha = acceptanceratio * dens_goal_Xnext / dens_prop_Xnext;
                double U = uni(gen);
                accept = (U < alpha);
            }
            X[i*max_dim + dim] = Xreal;
        }
    }
}
