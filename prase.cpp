#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// #include <pybind11/eigen.h>
#include <iostream>
#include <thread>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <stack>
#include <set>
#include <queue>
// #include <atomic>
#include <random>
#include <algorithm>

namespace py = pybind11;

class KG {
public:
    void insert_rel_triple(uint64_t, uint64_t, uint64_t);
    void insert_rel_inv_triple(uint64_t, uint64_t, uint64_t);
    void insert_attr_triple(uint64_t, uint64_t, uint64_t);
    void insert_attr_inv_triple(uint64_t, uint64_t, uint64_t);
    bool is_attribute(uint64_t);
    bool is_literal(uint64_t);
    void test();
    std::set<std::pair<uint64_t, uint64_t>>* get_rel_tail_pairs_ptr(uint64_t);
    std::set<std::pair<uint64_t, uint64_t>>* get_rel_head_pairs_ptr(uint64_t);
    std::set<uint64_t>& get_ent_set();
    double get_functionality(uint64_t);
    double get_inv_functionality(uint64_t);
private:
    std::set<uint64_t> ent_set;
    std::set<uint64_t> lite_set;
    std::set<uint64_t> attr_set;
    std::set<uint64_t> rel_set;
    std::unordered_map<uint64_t, std::set<std::pair<uint64_t, uint64_t>>> h_r_t_mp;
    std::unordered_map<uint64_t, std::set<std::pair<uint64_t, uint64_t>>> t_r_h_mp;
    std::unordered_map<uint64_t, double> functionality_mp;
    std::unordered_map<uint64_t, double> inv_functionality_mp;
    static std::set<std::pair<uint64_t, uint64_t>> EMPTY_PAIR_SET;
    static void insert_triple(std::unordered_map<uint64_t, std::set<std::pair<uint64_t, uint64_t>>>&, uint64_t, uint64_t, uint64_t);
    void init_functionalities();
};

std::set<std::pair<uint64_t, uint64_t>> KG::EMPTY_PAIR_SET = std::set<std::pair<uint64_t, uint64_t>>();

void KG::insert_triple(std::unordered_map<uint64_t, std::set<std::pair<uint64_t, uint64_t>>>& target, uint64_t head, uint64_t relation, uint64_t tail) {
    if (!target.count(head)) {
        target[head] = std::set<std::pair<uint64_t, uint64_t>>();
    }
    target[head].insert(std::make_pair(relation, tail));
}

void KG::insert_rel_triple(uint64_t head, uint64_t relation, uint64_t tail) {
    ent_set.insert(head);
    ent_set.insert(tail);
    rel_set.insert(relation);
    insert_triple(this -> h_r_t_mp, head, relation, tail);
    insert_triple(this -> t_r_h_mp, tail, relation, head);
}

void KG::insert_rel_inv_triple(uint64_t head, uint64_t relation_inv, uint64_t tail) {
    insert_rel_triple(tail, relation_inv, head);
}

void KG::insert_attr_triple(uint64_t entity, uint64_t attribute, uint64_t literal) {
    ent_set.insert(entity);
    lite_set.insert(literal);
    attr_set.insert(attribute);
    insert_triple(this -> h_r_t_mp, entity, attribute, literal);
    insert_triple(this -> t_r_h_mp, literal, attribute, entity);
}

void KG::insert_attr_inv_triple(uint64_t entity, uint64_t attribute_inv, uint64_t literal) {
    ent_set.insert(entity);
    lite_set.insert(literal);
    attr_set.insert(attribute_inv);
    insert_triple(this -> h_r_t_mp, literal, attribute_inv, entity);
    insert_triple(this -> t_r_h_mp, entity, attribute_inv, literal);
}

bool KG::is_attribute(uint64_t rel_id) {
    return !rel_set.count(rel_id);
}

bool KG::is_literal(uint64_t ent_id) {
    return !ent_set.count(ent_id);
}

std::set<std::pair<uint64_t, uint64_t>>* KG::get_rel_tail_pairs_ptr(uint64_t head_id) {
    if (!h_r_t_mp.count(head_id)) {
        return &EMPTY_PAIR_SET;
    }
    return &h_r_t_mp[head_id];
}

std::set<std::pair<uint64_t, uint64_t>>* KG::get_rel_head_pairs_ptr(uint64_t tail_id) {
    if (!t_r_h_mp.count(tail_id)) {
        return &EMPTY_PAIR_SET;
    }
    return &t_r_h_mp[tail_id];
}

std::set<uint64_t>& KG::get_ent_set() {
    return ent_set;
}

double KG::get_functionality(uint64_t rel_id) {
    double functionality = 0.0;
    if (functionality_mp.count(rel_id)) {
        functionality = functionality_mp[rel_id];
    }
    return functionality;
}

double KG::get_inv_functionality(uint64_t rel_id) {
    double inv_functionality = 0.0;
    if (inv_functionality_mp.count(rel_id)) {
        inv_functionality = inv_functionality_mp[rel_id];
    }
    return inv_functionality;
}

void KG::test() {
    init_functionalities();
    for (auto iter = functionality_mp.begin(); iter != functionality_mp.end(); ++iter) {
        std::cout<<"relation id: "<<iter -> first<<" functionality: "<<iter -> second<<std::endl;
    }
}

void KG::init_functionalities() {
    std::unordered_map<uint64_t, uint64_t> rel_id_triple_num_mp;
    std::unordered_map<uint64_t, std::set<uint64_t>> rel_id_head_set;
    std::unordered_map<uint64_t, std::set<uint64_t>> rel_id_tail_set;
    for (auto iter = h_r_t_mp.begin(); iter != h_r_t_mp.end(); ++iter) {
        uint64_t head = iter -> first;
        std::set<std::pair<uint64_t, uint64_t>> rel_tail_set = iter -> second;
        for (auto sub_iter = rel_tail_set.begin(); sub_iter != rel_tail_set.end(); ++sub_iter) {
             uint64_t relation = sub_iter -> first;
             uint64_t tail = sub_iter -> second;
             if (!rel_id_triple_num_mp.count(relation)) {
                 rel_id_triple_num_mp[relation] = 0;
             }
             if (!rel_id_head_set.count(relation)) {
                 rel_id_head_set[relation] = std::set<uint64_t>();
             }
             if (!rel_id_tail_set.count(relation)) {
                 rel_id_tail_set[relation] = std::set<uint64_t>();
             }
             ++rel_id_triple_num_mp[relation];
             rel_id_head_set[relation].insert(head);
             rel_id_tail_set[relation].insert(tail);
        }
    }
    for (uint64_t rel_id : rel_set) {
        uint64_t head_num = rel_id_head_set.count(rel_id) > 0 ? rel_id_head_set[rel_id].size() : 0;
        uint64_t tail_num = rel_id_tail_set.count(rel_id) > 0 ? rel_id_tail_set[rel_id].size() : 0;
        uint64_t total_num = rel_id_triple_num_mp.count(rel_id) ? rel_id_triple_num_mp[rel_id] : 0;
        double functionality = 0.0, inv_functionality = 0.0;
        if (total_num > 0) {
            functionality = ((double) head_num) / ((double) total_num);
            inv_functionality = ((double) tail_num) / ((double) total_num);
        }
        functionality_mp[rel_id] = functionality;
        inv_functionality_mp[rel_id] = inv_functionality;
    }

}

class PARISEquiv {
public:
    void insert_lite_equiv(uint64_t, uint64_t, double);
    void insert_ent_equiv(uint64_t, uint64_t, double);
    void insert_rel_equiv(uint64_t, uint64_t, double);
    double get_ent_equiv(KG*, uint64_t, KG*, uint64_t);
    double get_rel_equiv(uint64_t, uint64_t);
    std::unordered_map<uint64_t, double>* get_ent_cp_map_ptr(KG*, uint64_t);
    void insert_ongoing_rel_norm(std::unordered_map<uint64_t, double>&);
    void insert_ongoing_rel_deno(std::unordered_map<uint64_t, std::unordered_map<uint64_t, double>>&);
    void insert_ongoing_ent_eqv(std::unordered_map<uint64_t, std::unordered_map<uint64_t, double>>&);
    void update_rel_eqv(int);
    void update_ent_eqv();
    std::mutex rel_norm_lock;
    std::mutex rel_deno_lock;
    std::mutex ent_eqv_lock;
private:
    double get_entity_equiv(uint64_t, uint64_t);
    double get_literal_equiv(uint64_t, uint64_t);
    static double get_value_from_mp_mp(std::unordered_map<uint64_t, std::unordered_map<uint64_t, double>>&, uint64_t, uint64_t);
    static void insert_value_to_mp_mp(std::unordered_map<uint64_t, std::unordered_map<uint64_t, double>>&, uint64_t, uint64_t, double);
    static std::unordered_map<uint64_t, double> EMPTY_EQV_MAP;
    std::unordered_map<uint64_t, std::unordered_map<uint64_t, double>> ent_eqv_mp;
    std::unordered_map<uint64_t, std::unordered_map<uint64_t, double>> lite_eqv_mp;
    std::unordered_map<uint64_t, std::unordered_map<uint64_t, double>> rel_eqv_mp;
    std::unordered_map<uint64_t, double> ongoing_rel_norm;
    std::unordered_map<uint64_t, std::unordered_map<uint64_t, double>> ongoing_rel_deno;
    std::unordered_map<uint64_t, std::unordered_map<uint64_t, double>> ongoing_ent_eqv_mp;
};

std::unordered_map<uint64_t, double> PARISEquiv::EMPTY_EQV_MAP = std::unordered_map<uint64_t, double>();

void PARISEquiv::insert_lite_equiv(uint64_t lite_id_a, uint64_t lite_id_b, double prob) {
    insert_value_to_mp_mp(this -> lite_eqv_mp, lite_id_a, lite_id_b, prob);
}

void PARISEquiv::insert_ent_equiv(uint64_t ent_id_a, uint64_t ent_id_b, double prob) {
    insert_value_to_mp_mp(this -> ent_eqv_mp, ent_id_a, ent_id_b, prob);
}

void PARISEquiv::insert_rel_equiv(uint64_t rel_id_a, uint64_t rel_id_b, double prob) {
    insert_value_to_mp_mp(this -> rel_eqv_mp, rel_id_a, rel_id_b, prob);
}

double PARISEquiv::get_ent_equiv(KG* kg_a, uint64_t ent_id_a, KG* kg_b, uint64_t ent_id_b) {
    if (kg_a -> is_literal(ent_id_a)) {
        return get_entity_equiv(ent_id_a, ent_id_b);
    }
    return get_literal_equiv(ent_id_a, ent_id_b);
}

double PARISEquiv::get_rel_equiv(uint64_t rel_id, uint64_t rel_cp_id) {
    return get_value_from_mp_mp(rel_eqv_mp, rel_id, rel_cp_id);
}

std::unordered_map<uint64_t, double>* PARISEquiv::get_ent_cp_map_ptr(KG *source, uint64_t ent_id) {
    if (source ->is_literal(ent_id)) {
        if (lite_eqv_mp.count(ent_id)) {
            return &lite_eqv_mp[ent_id];
        }
        return &EMPTY_EQV_MAP;
    } else {
        if (ent_eqv_mp.count(ent_id)) {
            return &ent_eqv_mp[ent_id];
        }
        return &EMPTY_EQV_MAP;
    }
}

void PARISEquiv::insert_ongoing_rel_norm(std::unordered_map<uint64_t, double>& rel_norm_map) {
    for (auto iter = rel_norm_map.begin(); iter != rel_norm_map.end(); ++iter) {
        uint64_t relation = iter -> first;
        double factor = iter -> second;
        ongoing_rel_norm[relation] += factor;
    }
}

void PARISEquiv::insert_ongoing_rel_deno(std::unordered_map<uint64_t, std::unordered_map<uint64_t, double>>& rel_deno_map) {
    for (auto iter = rel_deno_map.begin(); iter != rel_deno_map.end(); ++iter) {
        uint64_t relation = iter -> first;
        std::unordered_map<uint64_t, double>& relation_cp_map = iter -> second;
        
        for (auto sub_iter = relation_cp_map.begin(); sub_iter != relation_cp_map.end(); ++sub_iter) {
            uint64_t relation_cp = sub_iter -> first;
            double factor = sub_iter -> second;
            
            if (!ongoing_rel_deno.count(relation)) {
                ongoing_rel_deno[relation] = std::unordered_map<uint64_t, double>();
            }
            ongoing_rel_deno[relation][relation_cp] += factor;
        }
    }
}

void PARISEquiv::insert_ongoing_ent_eqv(std::unordered_map<uint64_t, std::unordered_map<uint64_t, double>>& ent_eqv_result) {
    for (auto iter = ent_eqv_result.begin(); iter != ent_eqv_result.end(); ++iter) {
        uint64_t ent_id = iter -> first;
        std::unordered_map<uint64_t, double>& ent_cp_map = iter -> second;

        for (auto sub_iter = ent_cp_map.begin(); sub_iter != ent_cp_map.end(); ++sub_iter) {
            uint64_t end_cp_id = sub_iter -> first;
            double prob = sub_iter -> second;
            if (!ongoing_ent_eqv_mp.count(ent_id)) {
                ongoing_ent_eqv_mp[ent_id] = std::unordered_map<uint64_t, double>();
            }
            ongoing_ent_eqv_mp[ent_id][end_cp_id] = prob;
        }
    }
}

void PARISEquiv::update_rel_eqv(int norm_const) {
    rel_eqv_mp.clear();
    
    std::function<double(uint64_t)> get_rel_norm = [&](uint64_t rel_id) {
        double norm = 0.0;
        if (ongoing_rel_norm.count(rel_id)) {
            norm = ongoing_rel_norm[rel_id];
        }
        return norm;
    };

    for (auto iter = ongoing_rel_deno.begin(); iter != ongoing_rel_deno.end(); ++iter) {
        uint64_t relation = iter -> first;
        std::unordered_map<uint64_t, double>& rel_cp_map = iter -> second;
        double norm = get_rel_norm(relation) + norm_const;

        for (auto sub_iter = rel_cp_map.begin(); sub_iter != rel_cp_map.end(); ++sub_iter) {
            uint64_t relation_cp = sub_iter -> first;
            double prob = sub_iter -> second / norm;
            if (prob <= 0) {
                continue;
            }
            if (prob > 1.0) {
                prob = 1.0;
            }
            if (!rel_eqv_mp.count(relation)) {
                rel_eqv_mp[relation] = std::unordered_map<uint64_t, double>();
            }
            rel_eqv_mp[relation][relation_cp] = prob;
        }
    }
}

void PARISEquiv::update_ent_eqv() {

}

double PARISEquiv::get_value_from_mp_mp(std::unordered_map<uint64_t, std::unordered_map<uint64_t, double>> &mp, uint64_t id_a, uint64_t id_b) {
    if (!mp.count(id_a)) {
        return 0.0;
    }
    if (!mp[id_a].count(id_b)) {
        return 0.0;
    }
    return mp[id_a][id_b];
}

void PARISEquiv::insert_value_to_mp_mp(std::unordered_map<uint64_t, std::unordered_map<uint64_t, double>> &mp, uint64_t id_a, uint64_t id_b, double prob) {
    if (!mp.count(id_a)) {
        mp[id_a] = std::unordered_map<uint64_t, double>();
    }
    mp[id_a][id_b] = prob;
}

double PARISEquiv::get_entity_equiv(uint64_t entity_id_a, uint64_t entity_id_b) {
    return get_value_from_mp_mp(this -> ent_eqv_mp, entity_id_a, entity_id_b);
}

double PARISEquiv::get_literal_equiv(uint64_t literal_id_a, uint64_t literal_id_b) {
    return get_value_from_mp_mp(this -> lite_eqv_mp, literal_id_a, literal_id_b);
}

struct PARISParams {
    double ENT_EQV_THRESHOLD;
    double REL_EQV_THRESHOLD;
    double REL_EQV_FACTOR_THRESHOLD;
    double REL_INIT_EQV;
    double INIT_REL_EQV_PROB;
    double HIGH_CONF_THRESHOLD;
    double OUTPUT_THRESHOLD;
    double PENALTY_VALUE;
    double ENT_REGISTER_THRESHOLD;
    int INIT_ITERATION;
    int ENT_CANDIDATE_NUM;
    int SMOOTH_NORM;
    int THREAD_NUM;
    int MAX_THREAD_NUM;
    int MIN_THREAD_NUM;
    PARISParams();
};

PARISParams::PARISParams() {
    ENT_EQV_THRESHOLD = 0.1;
    REL_EQV_THRESHOLD = 0.1;
    REL_EQV_FACTOR_THRESHOLD = 0.05;
    REL_INIT_EQV = 0.1;
    INIT_REL_EQV_PROB = 0.1;
    HIGH_CONF_THRESHOLD = 0.9;
    OUTPUT_THRESHOLD = 0.1;
    PENALTY_VALUE = 1.01;
    ENT_REGISTER_THRESHOLD = 0.01;
    INIT_ITERATION = 2;
    ENT_CANDIDATE_NUM = 3;
    SMOOTH_NORM = 10;
    THREAD_NUM = std::thread::hardware_concurrency();
    MAX_THREAD_NUM = INT_MAX;
    MIN_THREAD_NUM = 1;
}


class PRModule {
public:
    PRModule(KG &, KG &);
    void insert_ent_eqv(uint64_t, uint64_t, double, bool);
    void insert_lite_eqv(uint64_t, uint64_t, double, bool);
    void insert_rel_eqv(uint64_t, uint64_t, double, bool);
    void enable_rel_init(bool);
private:
    int iteration;
    bool enable_relation_init;
    PARISEquiv* paris_eqv;
    PARISParams* paris_params;
    KG *kg_a, *kg_b;
    std::mutex queue_lock;
    std::unordered_map<uint64_t, std::unordered_map<uint64_t, double>> forced_eqv_mp;
    static void insert_value_to_mp_mp(std::unordered_map<uint64_t, std::unordered_map<uint64_t, double>>&, uint64_t, uint64_t, double);
    double get_filtered_prob(uint64_t, uint64_t, double);
    bool is_rel_init();
    static void one_iteration_one_way_per_thread(PRModule*, std::queue<uint64_t> &, KG*, KG*, bool);
    void one_iteration_one_way(std::queue<uint64_t> &, KG*, KG*, bool);
    void one_iteration();
};

PRModule::PRModule(KG &kg_a, KG &kg_b) {
    this -> kg_a = &kg_a;
    this -> kg_b = &kg_b;
    paris_eqv = new PARISEquiv();
    paris_params = new PARISParams();
    iteration = 0;
    enable_relation_init = true;
}

void PRModule::insert_value_to_mp_mp(std::unordered_map<uint64_t, std::unordered_map<uint64_t, double>> &mp, uint64_t id_a, uint64_t id_b, double prob) {
    if (!mp.count(id_a)) {
        mp[id_a] = std::unordered_map<uint64_t, double>();
    }
    mp[id_a][id_b] = prob;
}

void PRModule::insert_ent_eqv(uint64_t id_a, uint64_t id_b, double prob, bool forced) {
    paris_eqv -> insert_ent_equiv(id_a, id_b, prob);
    if (forced) {
        insert_value_to_mp_mp(this -> forced_eqv_mp, id_a, id_b, prob);
    }
}

void PRModule::insert_lite_eqv(uint64_t id_a, uint64_t id_b, double prob, bool forced) {
    paris_eqv -> insert_lite_equiv(id_a, id_b, prob);
    if (forced) {
        insert_value_to_mp_mp(this -> forced_eqv_mp, id_a, id_b, prob);
    }
}

void PRModule::insert_rel_eqv(uint64_t id_a, uint64_t id_b, double prob, bool forced) {
    paris_eqv -> insert_rel_equiv(id_a, id_b, prob);
    if (forced) {
        insert_value_to_mp_mp(this -> forced_eqv_mp, id_a, id_b, prob);
    }
}

void PRModule::enable_rel_init(bool flag) {
    enable_relation_init = flag;
}

double PRModule::get_filtered_prob(uint64_t id_a, uint64_t id_b, double prob) {
    if (forced_eqv_mp.count(id_a)) {
        if (forced_eqv_mp[id_a].count(id_b)) {
            prob = forced_eqv_mp[id_a][id_b];
        }
    }
    return prob;
}

bool PRModule::is_rel_init() {
    if (enable_relation_init) {
        if (iteration <= paris_params ->INIT_ITERATION) {
            return true;
        }
    }
    return false;
}

void PRModule::one_iteration_one_way_per_thread(PRModule* _this, std::queue<uint64_t>& ent_queue, KG* kg_l, KG* kg_r, bool ent_align) {
    uint64_t ent_id;
    std::unordered_map<uint64_t, std::unordered_map<uint64_t, double>> ent_eqv_result;
    std::unordered_map<uint64_t, double> ent_ongoing_eqv;
    std::unordered_map<uint64_t, double> rel_ongoing_norm_eqv;
    std::unordered_map<uint64_t, std::unordered_map<uint64_t, double>> rel_ongoing_deno_eqv;

    std::function<std::unordered_map<uint64_t, double>*(uint64_t)> get_cp_map_ptr = [&](uint64_t id) {
        std::unordered_map<uint64_t, double>* map_ptr;

        if (ent_eqv_result.count(id)) {
            map_ptr = &(ent_eqv_result[id]);
        } else {
            map_ptr = _this -> paris_eqv -> get_ent_cp_map_ptr(kg_l, id);
        }

        return map_ptr;
    };

    std::function<void(std::unordered_map<uint64_t, double>&, uint64_t, double)> register_ongoing_rel_eqv_deno 
    = [&](std::unordered_map<uint64_t, double>& mp, uint64_t cp_id, double prob) {
        if (!mp.count(cp_id)) {
            mp[cp_id] = 1.0;
        }
        mp[cp_id] *= (1.0 - prob);
    };

    std::function<void(uint64_t, uint64_t, uint64_t, double)> register_ongoing_ent_eqv = [&](uint64_t rel_id, uint64_t rel_cp_id, uint64_t head_cp_id, double tail_cp_eqv) {
        double rel_eqv_sub = _this -> paris_eqv ->get_rel_equiv (rel_id, rel_cp_id);
        double rel_eqv_sup = _this -> paris_eqv ->get_rel_equiv (rel_cp_id, rel_id);

        rel_eqv_sub /= _this -> paris_params -> PENALTY_VALUE;
        rel_eqv_sup /= _this -> paris_params -> PENALTY_VALUE;

        if (rel_eqv_sub < _this -> paris_params -> REL_EQV_THRESHOLD && rel_eqv_sup < _this -> paris_params -> REL_EQV_THRESHOLD) {
            if (_this -> is_rel_init()) {
                rel_eqv_sup = _this -> paris_params -> REL_EQV_THRESHOLD;
                rel_eqv_sub = _this -> paris_params -> REL_EQV_THRESHOLD;
            } else {
                return;
            }
        }

        double inv_functionality_l = kg_l -> get_inv_functionality(rel_id);
        double inv_functionality_r = kg_r -> get_inv_functionality(rel_cp_id);

        double factor = 1.0;

        if (inv_functionality_r >= 0.0 && rel_eqv_sub >= 0.0) {
            factor *= (1.0 - rel_eqv_sub * inv_functionality_r * tail_cp_eqv);
        }

        if (inv_functionality_l >= 0.0 && rel_eqv_sup >= 0.0) {
            factor *= (1.0 - rel_eqv_sup * inv_functionality_l * tail_cp_eqv);
        }

        if (1.0 - factor >= _this -> paris_params -> REL_EQV_FACTOR_THRESHOLD) {
            if (!ent_ongoing_eqv.count(head_cp_id)) {
                ent_ongoing_eqv[head_cp_id] = 1.0;
            }
            ent_ongoing_eqv[head_cp_id] *= factor;
        }

    };

    while (!ent_queue.empty()) {
        _this -> queue_lock.lock();
        if (!ent_queue.empty()) {
            ent_id = ent_queue.front();
            ent_queue.pop();
        } else {
            _this -> queue_lock.unlock();
            continue;
        }
        _this -> queue_lock.unlock();

        std::set<std::pair<uint64_t, uint64_t>>* rel_tail_pairs_ptr =  kg_l -> get_rel_tail_pairs_ptr(ent_id);
        std::unordered_map<uint64_t, double>* head_cp_ptr = get_cp_map_ptr(ent_id);

        for (auto iter = rel_tail_pairs_ptr -> begin(); iter != rel_tail_pairs_ptr -> end(); ++iter) {
            uint64_t relation = iter -> first;
            uint64_t tail = iter -> second;
            std::unordered_map<uint64_t, double>* tail_cp_ptr = get_cp_map_ptr(tail);

            double rel_ongoing_norm_factor = 1.0;
            std::unordered_map<uint64_t, double> rel_ongoing_deno_factor_map;

            for (auto tail_iter = tail_cp_ptr -> begin(); tail_iter != tail_cp_ptr -> end(); ++tail_iter) {
                uint64_t tail_cp = tail_iter -> first;
                double tail_eqv_prob = _this -> get_filtered_prob(tail, tail_cp, tail_iter -> second);

                if (tail_eqv_prob < _this -> paris_params -> ENT_EQV_THRESHOLD) {
                    continue;
                }

                for (auto head_iter = head_cp_ptr -> begin(); head_iter != head_cp_ptr -> end(); ++head_iter) {
                    uint64_t head_cp = head_iter -> first;
                    double head_eqv_prob = _this -> get_filtered_prob(ent_id, head_cp, head_iter -> second);
                    rel_ongoing_norm_factor *= (1.0 - head_eqv_prob * tail_eqv_prob);
                }

                std::set<std::pair<uint64_t, uint64_t>>* rel_head_pairs_ptr = kg_r -> get_rel_head_pairs_ptr(tail_cp);
                for (auto sub_iter = rel_tail_pairs_ptr -> begin(); sub_iter != rel_tail_pairs_ptr -> end(); ++sub_iter) {
                    uint64_t head_cp_candidate = sub_iter -> second;
                    if (kg_r -> is_literal(head_cp_candidate)) {
                        continue;
                    }

                    uint64_t relation_cp_candidate = sub_iter -> first;

                    if (head_cp_ptr -> count(head_cp_candidate)) {
                        double eqv_prob = _this -> get_filtered_prob(ent_id, head_cp_candidate, (*head_cp_ptr)[head_cp_candidate]);
                        register_ongoing_rel_eqv_deno(rel_ongoing_deno_factor_map, relation_cp_candidate, tail_eqv_prob * eqv_prob);
                    }
                    
                    if (ent_align) {
                        register_ongoing_ent_eqv(relation, relation_cp_candidate, head_cp_candidate, tail_eqv_prob);
                    }

                }
            }

            if (!rel_ongoing_norm_eqv.count(relation)) {
                rel_ongoing_norm_eqv[relation] = 0.0;
            }
            
            rel_ongoing_norm_eqv[relation] += 1.0 - rel_ongoing_norm_factor;

            for (auto deno_iter = rel_ongoing_deno_factor_map.begin(); deno_iter != rel_ongoing_deno_factor_map.end(); ++deno_iter) {
                uint64_t relation_cp_candidate = deno_iter -> first;
                double rel_eqv_deno = 1.0 - (deno_iter -> second);

                if (!rel_ongoing_deno_eqv.count(relation)) {
                    rel_ongoing_deno_eqv[relation] = std::unordered_map<uint64_t, double>();
                }
                rel_ongoing_deno_eqv[relation][relation_cp_candidate] += rel_eqv_deno;
            }

        }

        std::function<void()> update_ent_eqv = [&]() {
            std::stack<std::pair<uint64_t, double>> st1, st2;
            for (auto iter = ent_ongoing_eqv.begin(); iter != ent_ongoing_eqv.end(); ++iter) {
                uint64_t ent_cp_candidate = iter -> first;
                double ent_cp_eqv = _this -> get_filtered_prob(ent_id, ent_cp_candidate, 1.0 - (iter -> second));
                
                while (!st1.empty() && st1.top().second < ent_cp_eqv) {
                    st2.push(st1.top());
                    st1.pop();
                }
                
                if (st1.size() < _this -> paris_params -> ENT_CANDIDATE_NUM) {
                    st1.push(std::make_pair(ent_cp_candidate, ent_cp_eqv));
                }
                
                while (st1.size() < _this -> paris_params -> ENT_CANDIDATE_NUM && !st2.empty()) {
                    st1.push(st2.top());
                    st2.pop();
                }
            }
            
            while (!st1.empty()) {
                if (!ent_eqv_result.count(ent_id)) {
                    ent_eqv_result[ent_id] = std::unordered_map<uint64_t, double>();
                }
                ent_eqv_result[ent_id][st1.top().first] = st1.top().second;
                st1.pop();
            }

            ent_ongoing_eqv.clear();
        }; 
        
        if (ent_align) {
            update_ent_eqv();
        }
    }
    
    bool has_updated_rel_norm = false, has_updated_rel_deno = false, has_update_ent_eqv = !ent_align;

    while (!(has_updated_rel_norm && has_updated_rel_deno && has_update_ent_eqv)) {
        if (!has_updated_rel_norm) {
            if (_this -> paris_eqv -> rel_norm_lock.try_lock()) {
                has_updated_rel_norm = true;
                _this -> paris_eqv -> insert_ongoing_rel_norm(rel_ongoing_norm_eqv);
                _this -> paris_eqv -> rel_norm_lock.unlock();
            }
        } else if (!has_updated_rel_deno) {
            if (_this -> paris_eqv -> rel_deno_lock.try_lock()) {
                has_updated_rel_deno = true;
                _this -> paris_eqv -> insert_ongoing_rel_deno(rel_ongoing_deno_eqv);
                _this -> paris_eqv -> rel_deno_lock.unlock();
            }
        } else if (!has_update_ent_eqv) {
            if (_this -> paris_eqv -> ent_eqv_lock.try_lock()) {
                has_update_ent_eqv = true;
                _this -> paris_eqv -> insert_ongoing_ent_eqv(ent_eqv_result);
                _this -> paris_eqv -> ent_eqv_lock.unlock();
            }
        }
    }

}

void PRModule::one_iteration_one_way(std::queue<uint64_t> &q, KG* kg_l, KG* kg_r, bool ent_align) {
    int thread_num = std::max(std::min(paris_params -> THREAD_NUM, paris_params -> MAX_THREAD_NUM), paris_params -> MIN_THREAD_NUM);
    std::vector<std::thread> threads;

    for (int i = 0; i < thread_num; ++i) {
        threads.push_back(std::thread(&PRModule::one_iteration_one_way_per_thread, this, std::ref(q), kg_l, kg_r, ent_align));
    }

    for (int i = 0; i < thread_num; ++i) {
        threads[i].join();
    }

}

void PRModule::one_iteration() {
    ++iteration;

    std::queue<uint64_t> ent_queue;
    std::function<void(KG *)> set_ent_queue = [&](KG* curr_kg) {
        while (!ent_queue.empty()) {
            ent_queue.pop();
        }
        
        std::set<uint64_t>& ent_set = curr_kg -> get_ent_set();
        std::vector<uint64_t> ents = std::vector<uint64_t>(ent_set.begin(), ent_set.end());
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(ents.begin(), ents.end(), g);

        for (uint64_t ent : ents) {
            ent_queue.push(ent);
        }
    };

    set_ent_queue(kg_a);
    one_iteration_one_way(ent_queue, kg_a, kg_b, true);

    paris_eqv -> update_ent_eqv();

    set_ent_queue(kg_b);
    one_iteration_one_way(ent_queue, kg_b, kg_a, false);

    paris_eqv -> update_rel_eqv(paris_params -> SMOOTH_NORM);

}

PYBIND11_MODULE(prase, m)
{
    m.doc() = "Probabilistic Reasoning and Semantic Embedding";

    // m.def("add", &add, "A function which adds two numbers");

    py::class_<KG>(m, "KG").def(py::init()).def("insert_rel_triple", &KG::insert_rel_triple).def("test", &KG::test);
    py::class_<PRModule>(m, "PRModule").def(py::init<KG&, KG&>())
    .def("insert_ent_eqv", py::overload_cast<uint64_t, uint64_t, double, bool>(&PRModule::insert_ent_eqv));
}
