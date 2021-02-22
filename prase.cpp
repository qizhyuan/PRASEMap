#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// #include <pybind11/eigen.h>
#include <iostream>
#include <thread>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <set>
namespace py = pybind11;

// struct Entity {
//     uint64_t id;
//     std::string raw_name;
//     std::string name;
//     bool is_literal;
//     std::vector<std::pair<Relation, Entity>> rel_tail_list;
//     std::vector<std::pair<Relation, Entity>> rel_head_list;
//     Entity(uint64_t id, string name) : id(id), name(name), is_literal(false) {}
//     Entity(uint64_t id, string name, bool is_literal) : id(id), name(name), is_literal(is_literal) {}
// };

// struct Relation {
//     uint64_t id;
//     std::string raw_name;
//     std::string name;
//     bool is_attribute;
//     Relation* inv;
//     Relation(uint64_t id, std::string name) : id(id), name(name), is_attribute(false), inv(nullptr) {}
//     Relation(uint64_t id, std::string name, bool is_attribute) : id(id), name(name), is_attribute(is_attribute), inv(nullptr) {}
// };

class KG {
public:
    void insert_rel_triple(uint64_t, uint64_t, uint64_t);
    void insert_rel_inv_triple(uint64_t, uint64_t, uint64_t);
    void insert_attr_triple(uint64_t, uint64_t, uint64_t);
    void insert_attr_inv_triple(uint64_t, uint64_t, uint64_t);
private:
    std::set<uint64_t> ent_set;
    std::set<uint64_t> lite_set;
    std::set<uint64_t> attr_set;
    std::set<uint64_t> rel_set;
    std::unordered_map<uint64_t, std::set<std::pair<uint64_t, uint64_t>>> h_r_t_mp;
    std::unordered_map<uint64_t, std::set<std::pair<uint64_t, uint64_t>>> t_r_h_mp;
    std::unordered_map<uint64_t, double> functionality_mp;
    std::unordered_map<uint64_t, double> inv_functionality_mp;
    static void insert_triple(std::unordered_map<uint64_t, std::set<std::pair<uint64_t, uint64_t>>>&, uint64_t, uint64_t, uint64_t);
    void init_functionality();
};

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

PYBIND11_MODULE(prase, m)
{
    m.doc() = "Probabilistic Reasoning and Semantic Embedding";

    // m.def("add", &add, "A function which adds two numbers");

    py::class_<KG>(m, "KG").def(py::init());
}
