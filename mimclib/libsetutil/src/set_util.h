#ifndef __SET_UTIL_H__
#define __SET_UTIL_H__

typedef int int32;
typedef unsigned int uint32;
typedef unsigned short ind_t;

#ifdef __cplusplus
#include "var_list.h"
extern "C"{
#else
    typedef void* PProfitCalculator;
    typedef void* PVarSizeList;
    typedef unsigned int bool;
#endif

    PProfitCalculator GetMISCProfit(ind_t d, ind_t maxN,
                         const double *d_err_rates, const double *d_work_rates,
                         const double *s_g_rates, const double *s_g_bar_rates);
    PProfitCalculator GetAnisoProfit(ind_t d, const double *wE, const double *wW);

    double GetMinOuterProfit(const PVarSizeList, const PProfitCalculator profCalc);
    void CalculateSetProfit(const PVarSizeList, const PProfitCalculator profCalc,
                            double *log_error, double *log_work);
    void CheckAdmissibility(const PVarSizeList, ind_t d_start, ind_t d_end,
                            bool *admissible);
    void MakeProfitsAdmissible(const PVarSizeList, ind_t d_start, ind_t d_end,
                               double *pProfits);

    /* void GetLevelBoundaries(const PVarSizeList, const uint32 *levels, */
    /*                         uint32 levels_count, int32 *inner_bnd, */
    /*                         bool *inner_real_lvls); */
    /* void GetBoundaryInd(uint32 setSize, uint32 l, int32 i, */
    /*                     int32* sel, int32* inner_bnd, bool* bnd_ind); */


    PVarSizeList GetIndexSet(const PProfitCalculator profCalc,
                             double max_prof, double **p_profits);

    void GenTDSet(ind_t d, ind_t base, ind_t *td_set, uint32 count);
    void TensorGrid(ind_t d, ind_t base, const ind_t *m, ind_t* tensor_grid,
                    uint32 count);

    ind_t VarSizeList_max_dim(const PVarSizeList);
    ind_t VarSizeList_get(const PVarSizeList, uint32 i, ind_t* data, ind_t max_dim);
    ind_t VarSizeList_get_dim(const PVarSizeList, uint32 i);
    ind_t VarSizeList_get_active_dim(const PVarSizeList, uint32 i);

    uint32 VarSizeList_count(const PVarSizeList);
    PVarSizeList VarSizeList_sublist(const PVarSizeList, uint32* idx, uint32 _count);
    void VarSizeList_all_dim(const PVarSizeList, uint32 *dim, uint32 size);
    void VarSizeList_all_active_dim(const PVarSizeList, uint32 *active_dim, uint32 size);
    void VarSizeList_to_matrix(const PVarSizeList,
                               ind_t *ij, uint32 ij_size,
                               ind_t *data, uint32 data_size);
    int VarSizeList_find(const PVarSizeList, ind_t *j, ind_t *data,
                         ind_t size);
    PVarSizeList VarSizeList_from_matrix(const ind_t *sizes, uint32 count,
                                         const ind_t *j, uint32 j_size,
                                         const ind_t *data, uint32 data_size);

    void FreeProfitCalculator(PProfitCalculator profCalc);
    void FreeIndexSet(PVarSizeList);
    void FreeMemory(void **ind_set);
#ifdef __cplusplus
}
#endif

#endif    // __SET_UTIL_H__
