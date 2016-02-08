#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>
#include <boost/container/vector.hpp>
#include <boost/serialization/vector.hpp>
#include "../setutil/set_util.h"

#include "../setutil/cnpy/cnpy.h"

class Timer{
public:
    Timer() : start(std::chrono::high_resolution_clock::now()) {}
    double passed() {
        return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count();
    }
    //std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};

#define TIME(op, stmt) {Timer t; std::cout << op << std::flush; stmt; std::cout << "Took " << t.passed() << " sec. "<< std::endl;}

// std::ostream& operator<< (std::ostream& out, const set_t& v) {
//     out << "[";
//     size_t i=0;
//     for (auto itr=v.begin();itr!=v.end();itr++, i++){
//         out << *itr;
//         if (i+1 != v.size())
//             out << ", ";
//     }
//     out << "]";
//     return out;
// }

int main(){
    ind_t d = 1;
    double d_err_rates = 2*log(2);
    double d_work_rates = 1*log(2);
    double max_prof = 5.;
    double *p_profits;
    uint32 count;

    cnpy::NpyArray arr = cnpy::npy_load("setutil/g_rates.npy");
    double *s_g_rates = reinterpret_cast<double*>(arr.data);
    ind_t N = arr.shape[0];
    N = 3000;

    PProfitCalculator miscProf = GetMISCProfit(d, N, &d_err_rates,
                                               &d_work_rates,
                                               s_g_rates,
                                               s_g_rates);
    PVarSizeList pset;
    TIME("# Generating (prof=" << max_prof << ", N=" << N << ")... ",
         pset = GetIndexSet(miscProf, max_prof,  &p_profits));
    FreeProfitCalculator(miscProf);
    miscProf = GetMISCProfit(d, N, &d_err_rates, &d_work_rates,
                             s_g_rates, s_g_rates);

    count = pset->count();
    boost::container::vector<bool> admissible = boost::container::vector<bool>(count);
    std::vector<int32> inner_bnd = std::vector<int32>(count, -1);
    boost::container::vector<bool> inner_real_lvls = boost::container::vector<bool>(count);

    uint32 max_d = pset->max_dim();
    std::cout << "::Output >> Count=" << count << ", max_d=" << max_d << std::endl;

    bool* p_admissible = &admissible[0];
    // int32* p_inner_bnd = &inner_bnd[0];
    // bool* p_inner_real_lvls = &inner_real_lvls[0];
    TIME("# Checking admissibility...",
         CheckAdmissibility(pset, 0, max_d, p_admissible));
    TIME("# Making profits admissible...",
         MakeProfitsAdmissible(pset, 0, max_d, p_profits));
    double minProf;
    TIME("# Getting min outer profit...",
         minProf = GetMinOuterProfit(pset, miscProf));
    std::cout << "::Min profit: " << minProf << std::endl;
    // TIME("# Getting level boundaries...",
    //      GetLevelBoundaries(pset, &count, 1,
    //                         p_inner_bnd, p_inner_real_lvls))

    FreeProfitCalculator(miscProf);
    FreeIndexSet(pset);
    FreeMemory((void**)&p_profits);
    delete[] s_g_rates;

    std::cout << "Done!" << std::endl;
}
