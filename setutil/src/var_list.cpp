#include <cmath>
#include <map>
#include <set>
#include <limits>
#include <vector>
#include <list>
#include <assert.h>
#include <iostream>
#include "var_list.h"

#define DEBUG_ASSERT(x)
static const int SET_BASE = SparseMIndex::SET_BASE;

void VarSizeList::CheckAdmissibility(ind_t d_start, ind_t d_end,
                                     bool *admissible, uint32 size) const{
    assert(size >= this->count());
    for (unsigned int i=0;i<this->count();i++)
        admissible[i]=true;
    for (unsigned int i=0;i<this->count();i++){
        auto cur = this->get(i);
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


static double GetMinProfit(mul_ind_t cur, const VarSizeList& set,
                           std::vector<bool> &calc, double *pProfits,
                           ind_t d_start, ind_t d_end) {
    unsigned int iCur = set.find_ind(cur);
    //cur.resize(d_end, SET_BASE);
    for (unsigned int i=d_start;i<d_end;i++){
        cur.step(i, 1);
        unsigned int jj;
        if (set.find_ind(cur, jj)){
            if (!calc[jj])
                GetMinProfit(cur, set, calc, pProfits, d_start, d_end);
            pProfits[iCur] = std::min(pProfits[iCur], pProfits[jj]);
        }
        cur.step(i, -1);
    }
    calc[iCur] = true;
    return pProfits[iCur];
}

void VarSizeList::MakeProfitsAdmissible(ind_t d_start,
                                        ind_t d_end, double *pProfits,
                                        uint32 size) const {
    assert(size >= this->count());
    std::vector<bool> calc = std::vector<bool>(this->count(), false);
    GetMinProfit(mul_ind_t(), *this, calc, pProfits, d_start, d_end);
    // for (unsigned int i=0;i<set.count();i++)
    //     GetMinProfit(set.get(i), set, calc, pProfits, d_start, d_end);
    // Done!
}


double VarSizeList::GetMinOuterProfit(const PProfitCalculator profCalc) const {
    ind_t max_d = profCalc->MaxDim();//set.max_dim();
    std::vector<ind_t> bnd_neigh = std::vector<ind_t>(this->count(), 0);
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

    //------------- Calculate outer boundary
    double minProf = std::numeric_limits<double>::infinity();
    unsigned int bnd_count = 0;
    for (uint32 k=0;k<this->count();k++){
        if (bnd_neigh[k] < max_d) {
            // This is a boundary, check all outer indices that are
            // not in the set already
            auto cur = this->get(k);
            //std::cout << k << " / " << cur << " / " << profCalc->CalcLogProf(cur) << std::endl;
            double profit_cur = profCalc->CalcLogProf(cur);
            if (minProf < profit_cur)
                continue;  // No need to check further, profits only increase
            //cur.resize(max_d, SET_BASE);
            for (uint32 i=0;i<max_d;i++){
                cur.step(i, 1);
                if (!this->has_ind(cur) && minProf > profit_cur) {
                    double curProf = profCalc->CalcLogProf(cur);
                    minProf = std::min(minProf, curProf);
                    bnd_count++;
                }
                cur.step(i, -1);
            }
        }
    }
    return minProf;
}

void VarSizeList::CalculateSetProfit(const PProfitCalculator profCalc,
                                     double *log_error, double *log_work,
                                     uint32 size) const {
    assert(size >= this->count());
    for (size_t i=0;i<this->count();i++){
        profCalc->CalcLogEW(this->get(i), log_error[i], log_work[i]);
    }
}


void VarSizeList::GetLevelBoundaries(const uint32 *levels,
                                     uint32 levels_count,
                                     int32 *inner_bnd,
                                     bool *inner_real_lvls) {
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
//                     int32* sel, int32* inner_bnd, bool* bnd_ind){
//     // bnd_ind[self.sel][np.logical_and(self.inner_bnd >= self.i, inds < self.l)] = True
//     for (unsigned int j=0;j<setSize && j < l;j++){
//         bnd_ind[sel[j]] = (inner_bnd[j] >= i);
//     }
// }
