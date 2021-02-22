#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// #include <pybind11/eigen.h>
#include <iostream>
#include <thread>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <queue>
#include <atomic>
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
    std::set<std::pair<uint64_t, uint64_t>>& get_rel_tail_pair_set(uint64_t);
    std::set<std::pair<uint64_t, uint64_t>>& get_rel_head_pair_set(uint64_t);
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

std::set<std::pair<uint64_t, uint64_t>>& KG::get_rel_tail_pair_set(uint64_t head_id) {
    if (!h_r_t_mp.count(head_id)) {
        return this -> EMPTY_PAIR_SET;
    }
    return h_r_t_mp[head_id];
}

std::set<std::pair<uint64_t, uint64_t>>& KG::get_rel_head_pair_set(uint64_t tail_id) {
    if (!t_r_h_mp.count(tail_id)) {
        return this -> EMPTY_PAIR_SET;
    }
    return t_r_h_mp[tail_id];
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
    std::unordered_map<uint64_t, double>& get_ent_counterpart_map(KG*, uint64_t);
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
    std::unordered_map<uint64_t, std::unordered_map<uint64_t, double>> ongoing_rel_eqv_mp;
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

std::unordered_map<uint64_t, double>& PARISEquiv::get_ent_counterpart_map(KG *source, uint64_t ent_id) {
    if (source ->is_literal(ent_id)) {
        if (lite_eqv_mp.count(ent_id)) {
            return lite_eqv_mp[ent_id];
        }
        return EMPTY_EQV_MAP;
    } else {
        if (ent_eqv_mp.count(ent_id)) {
            return ent_eqv_mp[ent_id];
        }
        return EMPTY_EQV_MAP;
    }
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

class PRModule {
public:
    PRModule(KG &, KG &);
    void insert_ent_eqv(uint64_t, uint64_t, double, bool);
    void insert_lite_eqv(uint64_t, uint64_t, double, bool);
    void insert_rel_eqv(uint64_t, uint64_t, double, bool);
private:
    PARISEquiv* paris_eqv;
    KG *kg_a, *kg_b;
    std::mutex queue_lock;
    std::unordered_map<uint64_t, std::unordered_map<uint64_t, double>> forced_eqv_mp;
    static void insert_value_to_mp_mp(std::unordered_map<uint64_t, std::unordered_map<uint64_t, double>>&, uint64_t, uint64_t, double);
    void one_iteration_one_way(std::queue<uint64_t> &, KG*, KG*);
};

PRModule::PRModule(KG &kg_a, KG &kg_b) {
    this -> kg_a = &kg_a;
    this -> kg_b = &kg_b;
    paris_eqv = new PARISEquiv();
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

void PRModule::one_iteration_one_way(std::queue<uint64_t>& ent_queue, KG* kg_l, KG* kg_r) {
    uint64_t ent_id;
    std::unordered_map<uint64_t, std::unordered_map<uint64_t, double>> ent_eqv_result;
    std::unordered_map<uint64_t, double> ent_ongoing_eqv;
    while (!ent_queue.empty()) {
        queue_lock.lock();
        if (!ent_queue.empty()) {
            ent_id = ent_queue.front();
            ent_queue.pop();
        } else {
            queue_lock.unlock();
            continue;
        }
        queue_lock.unlock();
        std::set<std::pair<uint64_t, uint64_t>> rel_head_pairs =  kg_l -> get_rel_head_pair_set(ent_id);
        for (auto iter = rel_head_pairs.begin(); iter != rel_head_pairs.end(); ++iter) {
            uint64_t relation = iter -> first;
            uint64_t head = iter -> second;
            std::unordered_map<uint64_t, double> counterpart_map = paris_eqv -> get_ent_counterpart_map(kg_l, head);

        }
    }
    

}

PYBIND11_MODULE(prase, m)
{
    m.doc() = "Probabilistic Reasoning and Semantic Embedding";

    // m.def("add", &add, "A function which adds two numbers");

    py::class_<KG>(m, "KG").def(py::init()).def("insert_rel_triple", &KG::insert_rel_triple).def("test", &KG::test);
    py::class_<PRModule>(m, "PRModule").def(py::init<KG&, KG&>())
    .def("insert_ent_eqv", py::overload_cast<uint64_t, uint64_t, double, bool>(&PRModule::insert_ent_eqv));
}
