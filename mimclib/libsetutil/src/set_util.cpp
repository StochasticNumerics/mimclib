#include <cmath>
#include <map>
#include <set>
#include <limits>
#include <vector>
#include <list>
#include <assert.h>
#include <iostream>
#include "set_util.hpp"

#define DEBUG_ASSERT(x)
// static std::ostream& operator<<(std::ostream& out, const std::vector<ind_t>& v) {
//     out << "[";
//     size_t i=0;
//     for (i=0;i<v.size();i++){
//         out << v[i];
//         if (i+1 != v.size())
//             out << ", ";
//     }
//     out << "]";
//     return out;
// }

// #include <boost/functional/hash.hpp>
// std::size_t hash_ind(const mul_ind_t &ind){
//     return boost::hash_range(ind.begin(), ind.end());
// }

class MISCProfCalculator : public ProfitCalculator {
public:
    MISCProfCalculator(ind_t d, ind_t s,
                       const double *d_rates,
                       const double *s_err_rates) :
        d_rates(d_rates, d_rates+d),
        s_err_rates(s_err_rates, s_err_rates+s) { }

    double calc_log_prof(const mul_ind_t &cur) {
        check_ind(cur);
        double d_cont=0;
        double s_cont=0;
        unsigned int d = d_rates.size();
        auto mfun = [](unsigned int i) { return (i==0)?0:(i==1?1:(1+(1<<(i-1)))); };
        for (auto itr=cur.begin();itr!=cur.end();itr++){
            if (itr->ind < d)
                d_cont += (itr->value - SparseMIndex::SET_BASE)*d_rates[itr->ind];
            else{
                unsigned int M_1 = mfun(itr->value-SparseMIndex::SET_BASE);
                unsigned int M = mfun(1+itr->value-SparseMIndex::SET_BASE);
                double dM_1 = static_cast<double>(M_1);
                double dM = static_cast<double>(M);

                // log(dM - dM_1) is the work contribution
                s_cont += log(dM - dM_1) + dM_1*s_err_rates[itr->ind-d];
            }
        }
        return d_cont + s_cont;
    }

    ind_t max_dim() { return d_rates.size() + s_err_rates.size(); }
private:
    std::vector<double> d_rates;
    std::vector<double> s_err_rates;
};

// Total Degree profit calculator
class TDProfCalculator : public ProfitCalculator {
public:
    TDProfCalculator(ind_t d, const double *_weights) :
        weights(_weights, _weights+d) { }

    double calc_log_prof(const mul_ind_t &cur){
        check_ind(cur);
        double prof=0;
        for (auto itr=cur.begin();itr!=cur.end();itr++)
            prof += (itr->value-SparseMIndex::SET_BASE)*weights[itr->ind];
        return prof;
    }
    ind_t max_dim(){
        return weights.size();
    }
private:
    std::vector<double> weights;
};

// Full tensor profit calculator
class FTProfCalculator : public ProfitCalculator {
public:
    FTProfCalculator(ind_t d, const double *_weights) :
        weights(_weights, _weights+d) { }

    double calc_log_prof(const mul_ind_t &cur){
        check_ind(cur);
        double prof=0;
        for (auto itr=cur.begin();itr!=cur.end();itr++)
            prof = std::max(prof, (itr->value-SparseMIndex::SET_BASE)*weights[itr->ind]);
        return prof;
    }
    ind_t max_dim(){
        return weights.size();
    }
private:
    std::vector<double> weights;
};

ind_t* TensorGrid(ind_t d, uint32 td,
                  ind_t base, const ind_t *m, ind_t *cur, ind_t i,
                  ind_t* tensor_grid, uint32* pCount){
    if (i >= d){
        assert(td == 0);
        for (ind_t k=0;k<d;k++)
            *(tensor_grid++) = base+cur[k];
        (*pCount)--;
        return tensor_grid;
    }
    ind_t max = std::min(static_cast<ind_t>(m[i]-base), static_cast<ind_t>(td));
    for (cur[i]=(i==d-1)?td:0;
         cur[i]<=max;
         cur[i]++){
        tensor_grid = TensorGrid(d, td-cur[i], base, m, cur, i+1, tensor_grid, pCount);
        if (!(*pCount)) break;
    }
    return tensor_grid;
}

void TensorGrid(ind_t d, ind_t base,
                const ind_t *m, ind_t* tensor_grid, uint32 count){
    // Returns a tensorized product of m when i=1,2... m
    uint32 max_degree=0;
    for (ind_t i=0;i<d;i++) {
        assert(m[i] >= base);
        max_degree += m[i]-base+1;
    }

    ind_t cur[d];
    for (uint32 td=0;td<=max_degree;td++){
        tensor_grid=TensorGrid(d, td, base, m, cur, 0, tensor_grid, &count);
        if (!count) break;
    }
}

void FreeMemory(void **data)
{
    free(*data);
    *data = 0;
}


struct setprof_t{
    setprof_t(const mul_ind_t &i, double p): ind(i), profit(p), size(i.size()) {}
    setprof_t(const mul_ind_t &i, double p, ind_t _size): ind(i), profit(p), size(_size) {}

    mul_ind_t ind;
    double profit;
    ind_t size;
};
typedef std::list<setprof_t> ind_mul_ind_t;
bool compare_setprof(const setprof_t& first, const setprof_t& second)
{
    return first.profit < second.profit;
}

PVarSizeList GetIndexSet(PVarSizeList pRet,
                         const PProfitCalculator profCalc,
                         double max_prof,
                         double **p_profits) {
    ind_t max_d = profCalc->max_dim();
    ind_mul_ind_t ind_set;
    ind_set.push_back(setprof_t(mul_ind_t(), profCalc->calc_log_prof(mul_ind_t())));

    double cur_prof;
    ind_mul_ind_t::iterator itrCur = ind_set.begin();
    while (itrCur != ind_set.end()) {
        if (itrCur->size < max_d) {
            mul_ind_t cur_ind = itrCur->ind;
            ind_set.push_back(setprof_t(cur_ind, -1, itrCur->size+1));

            ind_t cur_size = itrCur->size;
            cur_ind.step(cur_size);
            while ((cur_prof=profCalc->calc_log_prof(cur_ind)) <= max_prof){
                ind_set.push_back(setprof_t(cur_ind, cur_prof));
                cur_ind.step(cur_size);
            }
            // if (added > 0)  // This assumes that dimensions are ordered!

        }
        // If this iterator has negative profit it means it's temporary and
        //  can be deleted safely (since all derivatives are already added)
        if (itrCur->profit < 0)
            itrCur = ind_set.erase(itrCur); // erase returns the next iterator
        else
            itrCur++;
    }

    ind_set.sort(compare_setprof);

    if (p_profits)
        *p_profits = static_cast<double*>(malloc(sizeof(double) * ind_set.size()));

    uint32 i=0;
    if (!pRet)
        pRet = new VarSizeList(ind_set.size());

    for (auto itr=ind_set.begin();itr!=ind_set.end();itr++){
        if (!pRet->has_ind(itr->ind))
            pRet->push_back(itr->ind);
        if (p_profits)
            (*p_profits)[i++] = itr->profit;
    }
    return pRet;
}


std::vector<ind_t> TDSet(ind_t d, uint32 count, unsigned int base){
    assert(d>=1 && count>=1);
    std::vector<ind_t> res;
    for (unsigned int k=0;k<d;k++){
        res.push_back(base);
    }
    std::vector<int> branch_point_prev;
    branch_point_prev.push_back(0);
    unsigned int cur_count = 1;   // Already added base
    unsigned int prev = 0;
    while (cur_count < count){
        std::vector<int> branch_point;
        for (unsigned int il=0;il<branch_point_prev.size();il++){
            for (unsigned int j=branch_point_prev[il];j<d;j++){
                branch_point.push_back(j);
                for (unsigned int k=0;k<d;k++){
                    res.push_back(res[(il+prev)*d+k] + (k==j));
                }
                cur_count++;
                if (cur_count >= count) break;
            }
            if (cur_count >= count) break;
        }
        prev += branch_point_prev.size();
        branch_point_prev = branch_point;
    }
    return res;
}

void GenTDSet(ind_t d, ind_t base, ind_t *td_set, uint32 count){
    std::vector<ind_t> ind_set = TDSet(d, count, base);
    assert(ind_set.size()/d==count);
    uint32 k=0;
    for (auto itr=ind_set.begin();itr!=ind_set.end();itr++){
        td_set[k++] = *itr;
    }
}


//////// ------------------------------------------------
/// C-Accessors to C++ methods
double GetMinOuterProfit(const PVarSizeList pset,
                         const PProfitCalculator profCalc){
    return pset->get_min_outer_profit(profCalc);
}

void CalculateSetProfit(const PVarSizeList pset,
                        const PProfitCalculator profCalc,
                        double *log_prof, uint32 size){
    pset->calc_set_profit(profCalc, log_prof, size);
}


void CheckAdmissibility(const PVarSizeList pset, ind_t d_start, ind_t d_end,
                        unsigned char *admissible){
    pset->check_admissibility(d_start, d_end, admissible, pset->count());
}

void MakeProfitsAdmissible(const PVarSizeList pset, ind_t d_start, ind_t d_end,
                           double *pProfits){
    pset->make_profits_admissible(d_start, d_end, pProfits, pset->count());
}

PVarSizeList VarSizeList_expand_set(const PVarSizeList pset,
                                    const double* error,
                                    const double* work,
                                    uint32 count,
                                    ind_t dimLookahead){
    PVarSizeList result = new VarSizeList();
    *result = pset->expand_set(error, work, count, dimLookahead);
    return result;
}

PVarSizeList VarSizeList_copy(const PVarSizeList lhs){
    if (lhs)
        return new VarSizeList(*lhs);
    PVarSizeList pset = new VarSizeList();
    return pset;
}
PVarSizeList VarSizeList_set_diff(const PVarSizeList lhs, const PVarSizeList rhs){
    PVarSizeList result = new VarSizeList();
    *result = lhs->set_diff(*rhs);
    return result;
}
PVarSizeList VarSizeList_set_union(const PVarSizeList lhs, const PVarSizeList rhs){
    PVarSizeList result = new VarSizeList();
    *result = lhs->set_union(*rhs);
    return result;
}


ind_t VarSizeList_get(const PVarSizeList pset, uint32 i, ind_t* data,
                      ind_t* j, ind_t size){
    const mul_ind_t& cur = pset->get(i);
    i=0;
    assert(cur.active() <= size);
    for (auto itr=cur.begin();itr!=cur.end();itr++, i++) {
        data[i] = itr->value;
        j[i] = itr->ind;
    }
    return cur.active();
}

uint32 VarSizeList_count(const PVarSizeList pset){
    return pset->count();
}

PVarSizeList VarSizeList_sublist(const PVarSizeList pset, uint32* idx, uint32 _count){
    return new VarSizeList(*pset, idx, _count);
}

ind_t VarSizeList_max_dim(const PVarSizeList pset){
    return pset->max_dim();
}

ind_t VarSizeList_get_dim(const PVarSizeList pset, uint32 i){
    return pset->get(i).size();
}

ind_t VarSizeList_get_active_dim(const PVarSizeList pset, uint32 i){
    return pset->get(i).active();
}

void VarSizeList_all_dim(const PVarSizeList pset, uint32 *dim, uint32 size){
    pset->all_dim(dim, size);
}

void VarSizeList_all_active_dim(const PVarSizeList pset, uint32 *active_dim, uint32 size){
    pset->all_active_dim(active_dim, size);
}


void VarSizeList_to_matrix(const PVarSizeList pset, ind_t *ij, uint32 ij_size,
                           ind_t *data, uint32 data_size){
    pset->to_matrix(ij, ij_size, data, data_size);
}

PProfitCalculator CreateMISCProfCalc(ind_t d, ind_t s, const double *d_w,
                                     const double *s_err_w){
    return new MISCProfCalculator(d, s, d_w, s_err_w);
}

PProfitCalculator CreateTDProfCalc(ind_t d, const double *w){
    return new TDProfCalculator(d, w);
}

PProfitCalculator CreateFTProfCalc(ind_t d, const double *w){
    return new FTProfCalculator(d, w);
}

void FreeProfitCalculator(PProfitCalculator profCalc){
    delete profCalc;
}

void FreeIndexSet(PVarSizeList pset){
    delete pset;
}

PVarSizeList VarSizeList_from_matrix(PVarSizeList pset,
                                     const ind_t *sizes, uint32 count,
                                     const ind_t *j, uint32 j_size,
                                     const ind_t *data, uint32 data_size){
    if (!pset)
        pset = new VarSizeList(count);
    uint32 total_size = 0;
    for (uint32 i=0;i<count;i++){
        total_size += sizes[i];
        assert(j_size >= total_size);
        assert(data_size >= total_size);
        //std::cout << "adding: " << mul_ind_t(j, data, sizes[i]) << "of size" << sizes[i] << std::endl;
        pset->push_back(mul_ind_t(j, data, sizes[i]));
        j += sizes[i];
        data += sizes[i];
    }
    return pset;
}

int VarSizeList_find(const PVarSizeList pset, ind_t *j, ind_t *data,
                     ind_t size){
    uint32 index;
    bool bfound = pset->find_ind(SparseMIndex(j, data, size), index);
    return bfound ? index:-1;
}

void VarSizeList_get_adaptive_order(const PVarSizeList pset,
                                    const double *error,
                                    const double *work,
                                    uint32 *adaptive_order,
                                    uint32 count,
                                    ind_t seedLookahead){
    pset->get_adaptive_order(error, work, adaptive_order, count, seedLookahead);
}

void VarSizeList_check_errors(const PVarSizeList pset, const double *errors, unsigned char* strange, uint32 count){
    pset->check_errors(errors, strange, count);
}

void VarSizeList_count_neighbors(const PVarSizeList pset, ind_t *out_neighbors, uint32 count){
    pset->count_neighbors(out_neighbors, count);
}

void VarSizeList_is_parent_of_admissible(const PVarSizeList pset, unsigned char *out, uint32 count){
    pset->is_parent_of_admissible(out, count);
}

double VarSizeList_estimate_bias(const PVarSizeList pset,
                                 const double *err_contributions,
                                 uint32 count,
                                 const double *rates, uint32 rates_size){
    return pset->estimate_bias(err_contributions, count,
                               rates, rates_size);
}
ind_t GetDefaultSetBase(){
    return SparseMIndex::SET_BASE;
}


PTree Tree_new(){
    return new Node(0);
}

unsigned char Tree_add_node(PTree tree, const double* value, uint32 count, double data, double eps){
    return tree->add_node(std::vector<double>(value, value+count), data, 0, eps);
}

unsigned char Tree_find(PTree tree, const double* value, uint32 count, double* data, unsigned char remove, double eps){
    return tree->find(std::vector<double>(value, value+count), *data, remove, 0, eps);
}

void Tree_free(PTree tree){
    delete tree;
}

void Tree_print(PTree tree){
    tree->print();
}
