#ifndef __VAR_LIST_H__
#define __VAR_LIST_H__

typedef int int32;
typedef unsigned int uint32;
typedef unsigned short ind_t;

#include <stdexcept>
#include <vector>
#include <map>
#include <list>
#include <algorithm>

class Node {
public:
    Node(double value) : m_value(value), m_data(-1),
                         m_is_leaf(false) {}

    bool add_node(const std::vector<double> &value, double data,
                  size_t depth=0, double eps=1e-14)
    {
        assert(depth <= value.size());
        if (depth == value.size()){
            m_data = data;
	    if (m_is_leaf)
	      return true;
	    m_is_leaf = true;
            return false;
        }
        auto mfun = [eps](const Node &node, const double &v)
	  { return std::abs(node.m_value - v) < eps? false : (node.m_value < v); };
        std::vector<Node>::iterator itr = std::lower_bound(m_nodes.begin(),
                                                           m_nodes.end(),
                                                           value[depth], mfun);
        if (itr != m_nodes.end() && std::abs(value[depth]-itr->m_value) < eps)
	  return itr->add_node(value, data, depth+1, eps);
        itr = m_nodes.insert(itr, Node(value[depth]));
        return itr->add_node(value, data, depth+1, eps);
    }

    bool find(const std::vector<double> &value, double &data,
              bool remove=false, size_t depth=0, double eps=1e-14) {
        if (depth == value.size()){
	  if (m_is_leaf){
	    data = m_data;
	    m_is_leaf = !remove;
	    return true;
	  }
	  return false;
        }
        auto mfun = [eps](const Node &node, const double &v)
	  { return std::abs(node.m_value - v) < eps? false : (node.m_value < v); };
        std::vector<Node>::iterator itr = std::lower_bound(m_nodes.begin(),
                                                           m_nodes.end(),
                                                           value[depth], mfun);

        if (itr != m_nodes.end() && std::abs(value[depth]-itr->m_value) < eps){
            return itr->find(value, data, remove, depth+1, eps);
        }
        return false;
    }

    void print(const std::string &pre=""){
        std::cout << pre << m_value;
        if (m_is_leaf)
	  std::cout << std::cout.setf(std::ios::fixed) << std::cout.precision(14) << "(" << m_data << ")";
        std::cout << std::endl;
        for (auto itr=m_nodes.begin();itr!=m_nodes.end();itr++){
	  itr->print(pre + "|-> ");
        }
    }

protected:
    double m_value;
    std::vector<Node> m_nodes;
    double m_data;
    bool m_is_leaf;
};

class SparseMIndex {
public:
    static const int SET_BASE = 0;
    struct Index
    {
    Index() : ind(0), value(SET_BASE) {}
    Index(ind_t _ind, ind_t _value) : ind(_ind), value(_value) {}
        ind_t ind;
        ind_t value;
    };
    typedef std::list<Index>::const_iterator const_iterator;
    typedef std::list<Index>::iterator iterator;

SparseMIndex() : m_max_size(0){ }
SparseMIndex(const SparseMIndex& rhs) :
    m_indices(rhs.m_indices), m_max_size(rhs.m_max_size){ }
SparseMIndex(const ind_t *ind, ind_t d) : m_max_size(0){
        for (ind_t i=0;i<d;i++)
            if (ind[i] != SET_BASE){
                assert(ind[i] > SET_BASE);
                m_indices.push_back(Index(i, ind[i]));
                m_max_size = std::max(m_max_size, i+1);
            }
    }

SparseMIndex(const ind_t *j, const ind_t *ind, ind_t d) : m_max_size(0){
        for (ind_t i=0;i<d;i++)
            if (ind[i] != SET_BASE){
                assert(ind[i] > SET_BASE);
                m_indices.push_back(Index(j[i], ind[i]));
                m_max_size = std::max(m_max_size, j[i]+1);
            }
    }

    inline ind_t get(ind_t i) const {
        const_iterator itr;
        if (!get_itr(i, itr)) return SET_BASE;
        return itr->value;
    }

    inline ind_t operator[] (ind_t i) const {
        return get(i);
    }
    void set(ind_t i, ind_t value){
        iterator itr;
        bool found = get_itr(i, itr);
        assert(value >= SET_BASE);
        if (!found){
            if (value > SET_BASE) {
                itr = m_indices.insert(itr, Index(i, value));
                m_max_size = std::max(m_max_size, i+1);
            } // Otherwise, just ignore the whole thing
        }
        else if (value > SET_BASE)
            itr->value = value;
        else {
            m_indices.erase(itr);
            update_size();
        }
    }

    void step(ind_t i, int step=1){
        iterator itr;
        bool found = get_itr(i, itr);
        if (!found){
            assert(step >= 0);
            itr = m_indices.insert(itr, Index(i, SET_BASE+step));
            m_max_size = std::max(m_max_size, i+1);
        }
        else
        {
            assert(static_cast<int>(itr->value) >= -step);
            if (itr->value+step > SET_BASE){
                itr->value+=step;
            }
            else
            {
                itr = m_indices.erase(itr);
                update_size();
            }
        }
    }

    ind_t size() const {
        return m_max_size;
    }

    ind_t active() const {
        return m_indices.size();
    }

    const_iterator begin() const { return m_indices.begin(); }
    const_iterator end() const { return m_indices.end(); }

    std::vector<ind_t> dense(ind_t dim) const{
        std::vector<ind_t> ret(dim, SET_BASE);
        for (auto itr=begin();itr!=end();itr++)
            ret[itr->ind] = itr->value;
        return ret;
    }

    bool operator<(const SparseMIndex& b) const {
        const SparseMIndex& a = *this;
        auto itr_b = b.begin();
        auto itr_a = a.begin();
        for (;
             itr_a != a.end() && itr_b != b.end();
             itr_a++, itr_b++){
            if (itr_a->ind == itr_b->ind) {
                if(itr_a->value == itr_b->value)
                    continue;
                else
                    return itr_a->value < itr_b->value;
            }
            // The indices are different, the one with the least
            // index is active
            return itr_a->ind > itr_b->ind;
            // if (itr_a->index < itr_b->index){
            //     // Means that b[itr_a->index] = SET_BASE
            //     return false;
            // }
            // if (itr_a->index > itr_b->index){
            //     // Means that a[itr_b->index] = SET_BASE
            //     return true;
            // }
        }

        if (itr_a == a.end() && itr_b == b.end())
            return false;   // Both indices are exhausted
        return itr_a == a.end();
    }

private:
    iterator begin() { return m_indices.begin(); }
    iterator end() { return m_indices.end(); }
    bool get_itr(ind_t i, iterator &ind_after) {
        for (auto itr=begin();itr!=end();itr++){
            if (itr->ind == i){
                ind_after = itr;
                return true;
            }
            else if (itr->ind > i) { // The indices are sorted
                ind_after = itr;
                return false;
            }
        }
        ind_after = end();
        return false;
    }
    bool get_itr(ind_t i, const_iterator &ind_after) const {
        for (auto itr=begin();itr!=end();itr++){
            if (itr->ind == i){
                ind_after = itr;
                return true;
            }
            else if (itr->ind > i) { // The indices are sorted
                ind_after = itr;
                return false;
            }
        }
        ind_after = end();
        return false;
    }
    void update_size(){
        m_max_size=0;
        for (auto itr=begin();itr!=end();itr++)
            m_max_size = std::max(m_max_size, itr->ind+1);
    }

    std::list<Index> m_indices;
    int m_max_size;
};
typedef SparseMIndex mul_ind_t;

class ProfitCalculator {
public:
    virtual ~ProfitCalculator(){}
    virtual double calc_log_prof(const mul_ind_t &ind)=0;
    virtual ind_t max_dim()=0;

    void check_ind(const mul_ind_t &ind){
        if (ind.size() > max_dim())
            throw std::runtime_error("Index too large for profit calculator");
    }
};

typedef ProfitCalculator* PProfitCalculator;

class VarSizeList {
public:
 VarSizeList(uint32 reserve=1) : m_max_dim(0) { m_ind_set.reserve(1); }
    VarSizeList(const VarSizeList &set, const uint32 *idx, uint32 _count) :
        m_max_dim(0)
    {
        for (uint32 i=0;i<_count;i++){
            if (idx[i] >= set.m_ind_set.size())
                throw std::out_of_range("Index larger than size");
            push_back(set.m_ind_set[idx[i]]);
        }
    }
 VarSizeList(const VarSizeList &set) : m_ind_set(set.m_ind_set) , m_ind_map(set.m_ind_map), m_max_dim(set.m_max_dim)
    { }

    ind_t max_dim() const {
        return m_max_dim;
        /* ind_t max_d = 0; */
        /* for (auto itr=m_ind_set.begin();itr!=m_ind_set.end();itr++) */
        /*     max_d = std::max(max_d, itr->size()); */
        /* return max_d; */
    }

    const mul_ind_t& get(uint32 i) const {
        assert(i<count());
        return m_ind_set[i];
    }

    ind_t get(uint32 i, ind_t j) const {
        return m_ind_set[i][j];
    }

    bool find_ind(const mul_ind_t& cur, uint32 &index) const{
        auto itr = m_ind_map.find(cur);
        if (itr == m_ind_map.end())
            return false;
        index = itr->second;
        return true;
    }

    uint32 find_ind(const mul_ind_t& cur) const {
        uint32 index;
        bool found = find_ind(cur, index);
        if (!found)
            throw std::out_of_range("Index not found!");
        return index;
    }

    bool has_ind(const mul_ind_t& cur) const {
        uint32 index;
        return find_ind(cur, index);
    }

    inline size_t count() const {
        return m_ind_set.size();
    }

    void all_dim(uint32 *dim, uint32 size) const{
        assert(size >= count());
        uint32 i=0;
        for (auto itr=m_ind_set.begin();itr!=m_ind_set.end();itr++)
            dim[i++] = itr->size();
    }

    void all_active_dim(uint32 *active_dim, uint32 size) const{
        assert(size >= count());
        uint32 i=0;
        for (auto itr=m_ind_set.begin();itr!=m_ind_set.end();itr++)
            active_dim[i++] = itr->active();
    }

    void to_matrix(ind_t *ij,
                   uint32 ij_size, // Asserts  2*count*np.sum(get_active_dim)
                   ind_t *data,
                   uint32 data_size // Asserts count*np.sum(get_active_dim)
        ) const {
        uint32 i=0;
        uint32 row=0;
        uint32 id=0;
        for (auto itr=m_ind_set.begin();itr!=m_ind_set.end();itr++)
        {
            for (auto ind_itr=itr->begin();ind_itr!=itr->end();ind_itr++){
                assert(id+1 <= data_size);
                assert(i+2 <= ij_size);
                data[id++] = ind_itr->value;
                ij[2*i] = row;
                ij[2*i+1] = ind_itr->ind;
                i++;
            }
            row++;
        }
    }

    void push_back(const mul_ind_t& ind){
        // WARNING: Does not check uniqueness
        if (this->has_ind(ind))
            throw std::runtime_error("Index already in set");
        m_ind_set.push_back(ind);
        m_ind_map[ind] = m_ind_set.size()-1;
        m_max_dim = std::max(m_max_dim, ind.size());
    }

    bool push_back_admiss(const mul_ind_t& ind){
        if (!this->has_ind(ind) && this->is_ind_admissible(ind)){
            push_back(ind);
            return true;
        }
        return false;
    }

    double get_min_outer_profit(const PProfitCalculator profCalc) const;
    void check_admissibility(ind_t d_start, ind_t d_end,
                            unsigned char *admissible, uint32 count) const;
    void make_profits_admissible(ind_t d_start, ind_t d_end,
                               double *pProfits, uint32 count) const;
    void calc_set_profit(const PProfitCalculator profCalc,
                         double *log_prof, uint32 count) const;
    void get_level_boundaries(const uint32 *levels, uint32 levels_count,
                              int32 *inner_bnd, unsigned char *inner_real_lvls) const;

#define DECLARE_ARR_ACCESSOR(NAME, TYPE) \
    void NAME(TYPE* out, size_t size) const;    \
    std::vector<TYPE> NAME() const {                                 \
        std::vector<TYPE> ret = std::vector<TYPE>(this->count());    \
        if (!this->count()) return ret; \
        NAME(&ret[0], ret.size()); return ret; }

    DECLARE_ARR_ACCESSOR(count_neighbors, ind_t);
    DECLARE_ARR_ACCESSOR(is_parent_of_admissible, unsigned char);  // std::vector<bool> is broken

    VarSizeList expand_set(const double *error,
                           const double *work,
                           uint32 count, ind_t dimLookahead) const;
    bool is_ind_admissible(const mul_ind_t& ind) const;
    VarSizeList set_diff(const VarSizeList& rhs) const;
    VarSizeList set_union(const VarSizeList& rhs) const;

    void get_adaptive_order(const double *error,
                            const double *work,
                            uint32 *adaptive_order,
                            uint32 count,
                            ind_t seedLookahead) const;

    void check_errors(const double *errors, unsigned char* strange, uint32 count) const;
    double estimate_bias(const double *err_contributions,
                         uint32 count, const double *rates, uint32 rates_size) const;
protected:
    typedef std::vector<mul_ind_t> ind_vector;
    ind_vector  m_ind_set;
    std::map<mul_ind_t, unsigned int> m_ind_map;
    ind_t m_max_dim;
};

typedef VarSizeList* PVarSizeList;


inline std::ostream& operator<< (std::ostream& out, const mul_ind_t& v) {
    out << "[";
    for (auto itr=v.begin();itr!=v.end();itr++){
        out << "(" << itr->ind << " -> " << itr->value << "), ";
    }
    out << "]";
    return out;
}

inline std::ostream& operator<< (std::ostream& out, const VarSizeList& v) {
    for (uint32 i=0;i<v.count();i++)
        out << v.get(i) << std::endl;
    return out;
}

#endif     //  __VAR_LIST_H__
