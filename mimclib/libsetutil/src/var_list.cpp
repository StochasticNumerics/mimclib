#include <cmath>
#include <map>
#include <set>
#include <limits>
#include <vector>
#include <list>
#include <assert.h>
#include <iostream>
#include <algorithm>
#include "var_list.hpp"

#define DEBUG_ASSERT(x)
static const int SET_BASE = SparseMIndex::SET_BASE;

template <typename T>
std::vector<uint32> argsort(const T &v, size_t size) {
    // initialize original index locations
    std::vector<uint32> idx(size);
    for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;
    // sort indexes based on comparing values in v
    std::sort(idx.begin(), idx.end(),
              [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
    return idx;
}
template <typename T>
std::vector<uint32> argsort(const std::vector<T>& v) {
    return argsort(v, v.size());
}

void VarSizeList::check_admissibility(ind_t d_start, ind_t d_end,
                                     unsigned char *admissible, uint32 size) const{
    assert(size >= this->count());
    for (unsigned int i=0;i<this->count();i++)
        admissible[i]=true;
    for (unsigned int i=0;i<this->count();i++){
        mul_ind_t cur = this->get(i);
        for (unsigned int j=d_start;
             j<d_end && j<cur.size() && admissible[i];
             j++){
            while(cur[j]>SET_BASE && admissible[i]){
                cur.step(j, -1);
                uint32 index;
                bool found = this->find_ind(cur, index);
                admissible[i] = found && admissible[index];
            }
            cur.set(j, this->get(i, j));
            DEBUG_ASSERT(cur.size() == this->get(i).size());
        }
    }
}


static double get_min_profit(mul_ind_t cur, const VarSizeList& set,
                           std::vector<bool> &calc, double *pProfits,
                           ind_t d_start, ind_t d_end) {
    unsigned int iCur = set.find_ind(cur);
    //cur.resize(d_end, SET_BASE);
    for (unsigned int i=d_start;i<d_end;i++){
        cur.step(i, 1);
        unsigned int jj;
        if (set.find_ind(cur, jj)){
            if (!calc[jj])
                get_min_profit(cur, set, calc, pProfits, d_start, d_end);
            pProfits[iCur] = std::min(pProfits[iCur], pProfits[jj]);
        }
        cur.step(i, -1);
    }
    calc[iCur] = true;
    return pProfits[iCur];
}

void VarSizeList::make_profits_admissible(ind_t d_start,
                                        ind_t d_end, double *pProfits,
                                        uint32 size) const {
    assert(size >= this->count());
    std::vector<bool> calc = std::vector<bool>(this->count(), false);
    get_min_profit(mul_ind_t(), *this, calc, pProfits, d_start, d_end);
    // for (unsigned int i=0;i<set.count();i++)
    //     get_min_profit(set.get(i), set, calc, pProfits, d_start, d_end);
    // Done!
}


// Returns the minimum profit on the outer set
double VarSizeList::get_min_outer_profit(const PProfitCalculator profCalc) const {
    ind_t max_d = profCalc->max_dim();//set.max_dim();
    std::vector<ind_t> bnd_neigh = this->count_neighbors();

    //------------- Calculate outer boundary
    double minProf = std::numeric_limits<double>::infinity();
    unsigned int bnd_count = 0;
    for (uint32 k=0;k<this->count();k++){
        if (bnd_neigh[k] < max_d) {
            // This is a boundary, check all outer indices that are
            // not in the set already
            auto cur = this->get(k);
            //std::cout << k << " / " << cur << " / " << profCalc->calc_log_prof(cur) << std::endl;
            double profit_cur = profCalc->calc_log_prof(cur);
            if (minProf < profit_cur)
                continue;  // No need to check further, profits only increase
            //cur.resize(max_d, SET_BASE);
            for (uint32 i=0;i<max_d;i++){
                cur.step(i, 1);
                if (!this->has_ind(cur) && minProf > profit_cur) {
                    double curProf = profCalc->calc_log_prof(cur);
                    minProf = std::min(minProf, curProf);
                    bnd_count++;
                }
                cur.step(i, -1);
            }
        }
    }
    return minProf;
}

void VarSizeList::calc_set_profit(const PProfitCalculator profCalc,
                                    double *log_prof,
                                    uint32 size) const {
    for (size_t i=0;i<this->count() && i < size;i++)
        log_prof[i] = profCalc->calc_log_prof(this->get(i));
}


void VarSizeList::get_level_boundaries(const uint32 *levels,
                                       uint32 levels_count,
                                       int32 *inner_bnd,
                                       unsigned char *inner_real_lvls) const {
    throw std::runtime_error("Must make it work with variables sets and also, the boundary of the axis is just the last element. Even if the axis is not covered!!!!!!!");
    /////// TODO: The following code does not work with variable size sets.
    const VarSizeList& set = *this;
    std::vector<ind_t> bnd_neigh = std::vector<ind_t>(set.count(), 0);
    uint32 start = 0;
    ind_t max_d = set.max_dim();
    for (unsigned int i=0;i<levels_count;i++){
        uint32 tmp = i==0 ? 0 : levels[i-1];   // Beginning of level set
        for (uint32 k=tmp;k<levels[i];k++){
            // Update Neighbors
            auto cur = set.get(k);
            for (unsigned int j=0;j<cur.size();j++){
                if (cur[j] == SET_BASE) continue;
                cur.step(j, -1);
                uint32 index;
                if (set.find_ind(cur, index))
                    bnd_neigh[index]++;
                cur.step(j, 1);
            }
        }
        // Copy to inner_bnd
        uint32 first = 0;
        for (uint32 k=levels[i];k>start;k--){
            if (bnd_neigh[k-1] < max_d){
                first = k-1;
                inner_bnd[k-1] = i;
            }
        }
        start = first;
    }
    // Check if levels are "real", i.e. they actually reduce the error
    inner_real_lvls[0] = true;
    int prev_lvl = 0;
    for (unsigned int i=1;i<levels_count;i++){
        inner_real_lvls[i] = false;
        // TODO: We can probably optimize so that we don't have to visit
        // all the indices
        for (unsigned int k=0;k<levels[i] && !inner_real_lvls[i];k++){
            // (inner_bnd[k] >= i)  (inner_bnd[k] >= prev_lvl)
            //      True               True                 -> False (Both levels have the same bnd)
            //      True               False                -> False (Current level adds to bnd)
            //      False              True                 -> True  (Current level removes from bnd)
            //      False              False                -> False (Current level does not change bnd)
            inner_real_lvls[i] = ((inner_bnd[k] < static_cast<int>(i)) &&
                                  (inner_bnd[k] >= prev_lvl));
        }
        if (inner_real_lvls[i])
            prev_lvl = i;
    }

    // Always include the last level
    inner_real_lvls[levels_count-1] = true;
}

// void GetBoundaryInd(uint32 setSize, uint32 l, int32 i,
//                     int32* sel, int32* inner_bnd, unsigned char* bnd_ind){
//     // bnd_ind[self.sel][np.logical_and(self.inner_bnd >= self.i, inds < self.l)] = True
//     for (unsigned int j=0;j<setSize && j < l;j++){
//         bnd_ind[sel[j]] = (inner_bnd[j] >= i);
//     }
// }

void VarSizeList::count_neighbors(ind_t* bnd_neigh, size_t size) const {
    assert(size >= this->count());
    for (uint32 k=0;k<this->count();k++) bnd_neigh[k]=0;

    for (uint32 k=0;k<this->count();k++){
        // Update Neighbors
        auto cur = this->get(k);
        for (unsigned int j=0;j<cur.size();j++){
            if (cur[j] == SET_BASE) continue;
            cur.step(j, -1);
            uint32 index;
            if (this->find_ind(cur, index))
                bnd_neigh[index]++;
            cur.step(j, 1);
            DEBUG_ASSERT(cur.size() == this->get(k).size());
        }
    }
}

uint32 add_children(const VarSizeList* pthis, uint32 k, VarSizeList &result,
    uint32 max_add){
    uint32 added = 0;
    auto cur = pthis->get(k);
    for (uint32 i=0;i<cur.size() && added < max_add;i++){
        cur.step(i, 1);
        if (!pthis->has_ind(cur) && pthis->is_ind_admissible(cur)
            && !result.has_ind(cur)){
            result.push_back(cur);
            added++;
        }
        cur.step(i, -1);
    }
    return added;
}

VarSizeList VarSizeList::expand_set(const double *error,
                                    const double *work,
                                    uint32 count,
                                    ind_t seedLookahead) const{
    // Sorted ind lists the indices in order of expansion (of decreasing profit).
    VarSizeList result;
    assert(count == this->count());
    std::vector<double> profits(count);
    for (size_t i=0;i<profits.size();i++)
        profits[i] = log(work[i]) - log(fabs(error[i]));

    std::vector<uint32> sorted_ind = argsort(profits);
    std::vector<ind_t> bnd_neigh = this->count_neighbors();

#define ADD_IND(ind) if (!this->has_ind(ind)) result.push_back(ind)
    //------------- Calculate outer boundary
    ind_t max_dim = this->max_dim();
    uint32 added = 0;
    for (uint32 j=0;j<count && added == 0;j++){
        uint32 k = sorted_ind[j];   // Start from the least -logprofit
        if (bnd_neigh[k] < max_dim) {
            // This is a boundary, check all outer indices that are
            // not in the set already
            added += add_children(this, k, result, max_dim);
        }
    }

    // Expand set where on boundaries where there are exactly errors=0
    // The idea is that these indices might be a fluke and it is better
    // to explore these indices
    std::vector<uint32> sorted_work_ind = argsort(work, count);
    added = 0;
    for (uint32 j=0;j<count && added == 0;j++){
        uint32 k = sorted_work_ind[j];   // Start from the least work
        if (bnd_neigh[k] < max_dim && error[k] == 0.0) {
            added += add_children(this, k, result, 1);
        }
    }

    // Now add at least seedLookahead + base element
    mul_ind_t cur;  // Set all to zeros
    ADD_IND(cur);
    if (!seedLookahead)
        return result;

    if (max_dim < seedLookahead)
    {
        uint32 diff = static_cast<uint32>(seedLookahead-max_dim);
        for (uint32 i=max_dim;i<diff;i++) {
            cur.step(i, 1);
            // No need to check for admissibility since new dimension only
            // require the zero element and we already made sure it is added
            ADD_IND(cur);
            max_dim = std::max(max_dim, cur.size());
            cur.step(i, -1);
        }
    }
    for (ind_t i=0;i<seedLookahead;i++){
        // Count number of elements with exactly size=max_dim-i
        uint32 count = 0;
        for (auto itr=m_ind_set.begin();itr!=m_ind_set.end();itr++)
            count += (itr->size() == (max_dim - i));
        if (count > 1){  // Has more than one element
            // Add missing seeds
            for (int j=0;j<seedLookahead-i;j++) {
                cur.step(max_dim+j, 1);
                // No need to check for admissibility since new dimension only
                // require the zero element and we already made sure it is added
                ADD_IND(cur);
                cur.step(max_dim+j, -1);
            }
            break;
        }
    }
#undef ADD_CUR
    return result;
}

bool VarSizeList::is_ind_admissible(const mul_ind_t& ind) const{
    mul_ind_t cur = ind;
    for (unsigned int j=0;
         j<cur.size();
         j++){
        while(cur[j]>SET_BASE){
            cur.step(j, -1);
            uint32 index;
            if (!this->find_ind(cur, index))
                return false;
        }
        cur.set(j, ind.get(j));
        DEBUG_ASSERT(cur.size() == ind.size());
    }
    return true;
}

VarSizeList VarSizeList::set_diff(const VarSizeList& rhs) const {
    VarSizeList result = VarSizeList();
    for (auto itr=this->m_ind_set.begin();itr!=this->m_ind_set.end();itr++)
        if (!rhs.has_ind(*itr))
            result.push_back(*itr);
    return result;
}

VarSizeList VarSizeList::set_union(const VarSizeList& rhs) const{
    VarSizeList result = VarSizeList(*this);
    for (auto itr=rhs.m_ind_set.begin();itr!=rhs.m_ind_set.end();itr++)
        if (!result.has_ind(*itr))
            result.push_back(*itr);
    return result;
}


void VarSizeList::get_adaptive_order(const double *error,
                                     const double *work,
                                     uint32 *adaptive_order,
                                     uint32 count,
                                     ind_t seedLookahead) const
{
    // Start with an empty index
    // TODO: Can be heavily optimized
    for (uint32 i=0;i<count;i++)
        adaptive_order[i] = std::numeric_limits<uint32>::max();

    VarSizeList curList;
    uint32 cur_order = 1;
    std::vector<double> error_in_set;
    std::vector<double> work_in_set;
    while (1){
        VarSizeList newList = curList.expand_set(&error_in_set[0],
                                                 &work_in_set[0],
                                                 curList.count(),
                                                 seedLookahead);
        bool all_found = true;
        for (auto itr=newList.m_ind_set.begin();itr!=newList.m_ind_set.end();itr++){
            uint32 ii;
            if (!this->find_ind(*itr, ii)){
                all_found = false;
                break;
            }
            adaptive_order[ii] = cur_order;
            error_in_set.push_back(error[ii]);
            work_in_set.push_back(work[ii]);
        }
        if (!all_found)
            break;
        curList = curList.set_union(newList);
        cur_order++;
    }
}

void VarSizeList::check_errors(const double *errors, unsigned char* strange, uint32 count) const{
    assert(count == this->count());
    for (uint32 k=0;k<this->count();k++){
        // Update Neighbors
        strange[k] = false;
        if (errors[k] == 0)
            continue;
        auto cur = this->get(k);
        for (unsigned int j=0;j<cur.size();j++){
            if (cur[j] == SET_BASE) continue;
            cur.step(j, -1);
            uint32 index;
            if (this->find_ind(cur, index) && errors[index] == 0){
                strange[k] = true;
            }
            cur.step(j, 1);
            DEBUG_ASSERT(cur.size() == this->get(k).size());
        }
    }
}

void VarSizeList::is_parent_of_admissible(unsigned char* pout, size_t size) const{
    ind_t max_d = this->max_dim();
    std::vector<ind_t> bnd_neigh = this->count_neighbors();
    //------------- Calculate outer boundary
    for (uint32 k=0;k<this->count();k++){
        pout[k] = false;
        if (bnd_neigh[k] < max_d) {
            // This is a boundary, check all outer indices that are not in the set already
            auto cur = this->get(k);
            for (uint32 i=0;i<max_d;i++){
                cur.step(i, 1);
                if (!this->is_ind_admissible(cur)) {
                    pout[k] = true;
                    break;
                }
                cur.step(i, -1);
            }
        }
    }
}

double VarSizeList::estimate_bias(const double *err_contributions,
                                  uint32 count,
                                  const double *rates, uint32 rates_size) const {
    ind_t max_d = this->max_dim();
    assert(count >= this->count && rates_size >= max_d);
    std::vector<ind_t> bnd_neigh = this->count_neighbors();

    std::map<mul_ind_t, double> map_contrib;
    //------------- Calculate outer boundary
    for (uint32 k=0;k<this->count();k++){
        if (bnd_neigh[k] < max_d) {
            // This is a boundary, check all outer indices that are
            // not in the set already
            auto cur = this->get(k);
            for (uint32 i=0;i<max_d;i++){
                cur.step(i, 1);
                if (!this->has_ind(cur) && this->is_ind_admissible(cur)) {
                    double prev = std::numeric_limits<double>::infinity();
                    auto itr = map_contrib.find(cur);
                    if (itr != map_contrib.end()) prev = std::abs(itr->second);
                    //map_contrib[cur] = std::min(prev, rates[i]*err_contributions[k]);
                    if (prev > std::abs(rates[i]*err_contributions[k]))
                        map_contrib[cur] = rates[i]*err_contributions[k];
                }
                cur.step(i, -1);
            }
        }
    }
    double bias=0;
    for (auto itr=map_contrib.begin();itr!=map_contrib.end();itr++){
        bias += itr->second;
    }
    return bias;
}
