import json
import os
import random


def get_abbr_name(name):
    value = name
    if "/" in value:
        value = value.split(sep="/")[-1].strip()
    return value


def get_random_candidate_list(ent_list, num=5):
    if len(ent_list) <= num:
        return ent_list
    if num < len(ent_list) <= num * 10:
        return random.sample(ent_list, num)
    candidate_num = int(len(ent_list) * 0.15)
    new_ent_list = ent_list[:candidate_num]
    return random.sample(new_ent_list, num)


def generate_node_and_edge_sets(kg, cent_node, hop_num):
    triple_set = set()
    entity_set = set()
    entity_set.add(cent_node)
    newly_added_ent_set = entity_set.copy()

    for _ in range(hop_num):
        next_newly_added_ent_set = set()
        for ent in newly_added_ent_set:
            for (rel, tail) in kg.get_rel_ent_id_tuples_by_ent(ent):
                if tail not in entity_set:
                    entity_set.add(tail)
                    next_newly_added_ent_set.add(tail)
        newly_added_ent_set = next_newly_added_ent_set

    for ent in entity_set:
        for (rel, tail) in kg.get_rel_ent_id_tuples_by_ent(ent):
            if tail in entity_set:
                triple_set.add((ent, rel, tail))

    new_triple_set = set()
    for (h, r, t) in triple_set:
        if kg.is_inv_rel(r):
            new_triple_set.add((t, kg.get_inv_id(r), h))
        else:
            new_triple_set.add((h, r, t))

    return entity_set, new_triple_set


def generate_node_info_list(kg, entity_set, entity_id_dict, group=None):
    new_index = len(entity_id_dict)
    node_info_list = list()

    for ent in entity_set:
        entity_id_dict[ent] = str(new_index)
        ent_info_dict = dict()
        ent_info_dict["nodeId"] = str(new_index)
        ent_info_dict["nodeName"] = get_abbr_name(kg.get_ent_name_by_id(ent))
        if group is not None:
            ent_info_dict["group"] = group
        ent_info_dict["attributes"] = dict()
        for (attr, lite) in kg.get_attr_lite_id_tuples_by_ent(ent):
            attribute = get_abbr_name(kg.get_attr_name_by_id(attr))
            literal = kg.get_lite_name_by_id(lite)
            if len(literal) >= 30:
                literal = literal[:30] + "..."
            ent_info_dict["attributes"][attribute] = literal
        node_info_list.append(ent_info_dict)
        new_index += 1

    return node_info_list, entity_id_dict


def generate_pairs_for_correction(kgs):
    correction_pair_candidates1 = kgs.se_feedback_pairs.copy()
    if len(correction_pair_candidates1) >= 500:
        correction_pair_candidates1 = random.sample(correction_pair_candidates1, 500)

    correction_pair_candidates2 = set()
    aligned_pairs = kgs.get_ent_align_ids_result()
    for (ent, ent_cp, prob) in aligned_pairs:
        if prob < 0.5:
            correction_pair_candidates2.add((ent, ent_cp))

    if len(correction_pair_candidates2) >= 500:
        correction_pair_candidates2 = random.sample(correction_pair_candidates2, 500)
    else:
        if len(aligned_pairs) < 500:
            correction_pair_candidates2 = aligned_pairs.copy()
        else:
            correction_pair_candidates2 = random.sample(aligned_pairs, 500)
    correction_pair_candidates1 = set(correction_pair_candidates1)
    correction_pair_candidates2 = set(correction_pair_candidates2)

    correction_pair_candidates = list(correction_pair_candidates1 | correction_pair_candidates2)
    correction_index = range(len(correction_pair_candidates))

    if len(correction_index) > 200:
        correction_index = random.sample(correction_index, 200)

    correction_list = [(correction_pair_candidates[i][0], correction_pair_candidates[i][1]) for i in correction_index]
    result_list = list()
    # result_dict["markingNum"] = len(correction_list)

    def generate_node_info(kg, node_id):
        node_info_dict = dict()
        name = kg.get_ent_name_by_id(node_id)
        node_info_dict["nodeName"] = name
        node_info_dict["relations"] = dict()
        node_info_dict["attributes"] = dict()
        for (attr, lite) in kg.get_attr_lite_id_tuples_by_ent(node_id):
            attribute = get_abbr_name(kg.get_attr_name_by_id(attr))
            literal = kg.get_lite_name_by_id(lite)
            if len(literal) >= 30:
                literal = literal[:30] + "..."
            node_info_dict["attributes"][attribute] = literal

        for (rel, tail) in kg.get_rel_ent_id_tuples_by_ent(node_id):
            relation = get_abbr_name(kg.get_rel_name_by_id(rel))
            entity = get_abbr_name(kg.get_ent_name_by_id(tail))
            node_info_dict["relations"][relation] = entity
        return node_info_dict

    for (ent, ent_cp) in correction_list:
        node_info1, node_info2 = generate_node_info(kgs.kg1, ent), generate_node_info(kgs.kg2, ent_cp)
        item_dict = {"KG1": node_info1, "KG2": node_info2}
        result_list.append(item_dict)

    return result_list


def generate_edge_info_list(kg, triple_set, entity_id_dict):
    edge_info_list = list()

    for (h, r, t) in triple_set:
        edge_info_dict = dict()
        h_id = entity_id_dict[h]
        t_id = entity_id_dict[t]
        r_name = get_abbr_name(kg.get_rel_name_by_id(r))
        edge_info_dict["sourceNodeId"] = h_id
        edge_info_dict["targetNodeId"] = t_id
        edge_info_dict["edgeName"] = r_name
        edge_info_list.append(edge_info_dict)

    return edge_info_list


def construct_single_kg_demo_file(kgs, save_path, hop=1, kg1_name="KG1", kg2_name="KG2"):
    base, path = os.path.split(save_path)
    if not os.path.exists(base):
        os.makedirs(base)

    demo_dict = dict()

    def calculate_score(kg, ent_id):
        tuples = kg.get_rel_ent_id_tuples_by_ent(ent_id)
        return len(tuples)

    kg1_ent_list = list(kgs.kg1.get_ent_id_set())
    kg2_ent_list = list(kgs.kg2.get_ent_id_set())
    kg1_ent_list.sort(key=lambda x: calculate_score(kgs.kg1, x), reverse=True)
    kg2_ent_list.sort(key=lambda x: calculate_score(kgs.kg2, x), reverse=True)

    kg1_ent_list = get_random_candidate_list(kg1_ent_list)
    kg2_ent_list = get_random_candidate_list(kg2_ent_list)

    demo_dict["KG1"] = dict()
    demo_dict["KG2"] = dict()

    def construction(dict_file, kg, kg_index, ent_list, hop_num, kg_name):
        dict_file[kg_index]["info"] = dict()
        dict_file[kg_index]["info"]["kgName"] = kg_name
        dict_file[kg_index]["info"]["entNum"] = len(kg.get_ent_id_set())
        dict_file[kg_index]["info"]["relNum"] = int(len(kg.get_rel_id_set()) / 2)
        dict_file[kg_index]["info"]["attrNum"] = int(len(kg.get_attr_id_set()) / 2)
        dict_file[kg_index]["info"]["liteNum"] = int(len(kg.get_lite_id_set()) / 2)
        dict_file[kg_index]["info"]["relTripleNum"] = int(len(kg.get_relation_id_triples()) / 2)
        dict_file[kg_index]["info"]["attrTripleNum"] = int(len(kg.get_attribute_id_triples()) / 2)
        dict_file[kg_index]["subGraphs"] = dict()
        dict_file[kg_index]["subGraphs"]["subGraphNum"] = len(ent_list)
        for i in range(len(ent_list)):
            dict_file[kg_index]["subGraphs"]["subGraph-" + str(i + 1)] = dict()
            sub_graph_dict = dict_file[kg_index]["subGraphs"]["subGraph-" + str(i + 1)]
            cent_index = ent_list[i]
            cent_name = get_abbr_name(kg.get_ent_name_by_id(cent_index))
            sub_graph_dict["centerNodeName"] = cent_name
            sub_graph_dict["centerNodeId"] = None
            sub_graph_dict["nodes"] = list()
            sub_graph_dict["edges"] = list()
            entity_set, triple_set = generate_node_and_edge_sets(kg, cent_index, hop_num)

            entity_new_id_dict = dict()
            ent_info_list, entity_new_id_dict = generate_node_info_list(kg, entity_set, entity_new_id_dict)

            edge_info_list = generate_edge_info_list(kg, triple_set, entity_new_id_dict)

            sub_graph_dict["nodes"] = ent_info_list
            sub_graph_dict["edges"] = edge_info_list

            sub_graph_dict["centerNodeId"] = entity_new_id_dict[cent_index]

    construction(demo_dict, kgs.kg1, "KG1", kg1_ent_list, hop, kg1_name)
    construction(demo_dict, kgs.kg2, "KG2", kg2_ent_list, hop, kg2_name)

    with open(save_path, "w", encoding="utf8") as f:
        json.dump(demo_dict, f, indent=4)


def construct_kg_mappings_demo_file(kgs, save_path, hop=1, kg1_name="KG1", kg2_name="KG2"):
    base, path = os.path.split(save_path)
    if not os.path.exists(base):
        os.makedirs(base)

    demo_dict = dict()

    ent_mappings = kgs.get_ent_align_ids_result()
    ent_mappings = list(ent_mappings)

    ent_mappings_dict = dict()
    ent_mappings_prob = dict()

    for (ent, ent_cp, prob) in ent_mappings:
        ent_mappings_dict[ent] = ent_cp
        ent_mappings_prob[(ent, ent_cp)] = prob

    def calculate_score(x):
        ent_l, ent_r, prob = x[0], x[1], x[2]
        deg1 = len(kgs.kg1.get_rel_ent_id_tuples_by_ent(ent_l))
        deg2 = len(kgs.kg2.get_rel_ent_id_tuples_by_ent(ent_r))
        score = (deg1 + deg2) * prob
        return score

    ent_mappings.sort(key=lambda x: calculate_score(x), reverse=True)

    ent_pairs = get_random_candidate_list(ent_mappings)
    demo_dict["info"] = dict()

    def construction(dict_file, hop_num):
        dict_file["info"] = dict()
        dict_file["info"]["mappingNum"] = len(kgs.get_ent_align_ids_result())
        dict_file["subGraphs"] = dict()
        dict_file["subGraphs"]["subGraphNum"] = len(ent_pairs)
        for i in range(len(ent_pairs)):
            dict_file["subGraphs"]["subGraph-" + str(i + 1)] = dict()
            sub_graph_dict = dict_file["subGraphs"]["subGraph-" + str(i + 1)]
            cent_index_l, cent_index_r = ent_pairs[i][0], ent_pairs[i][1]

            sub_graph_dict["centerNodeName-1"] = get_abbr_name(kgs.kg1.get_ent_name_by_id(cent_index_l))
            sub_graph_dict["centerNodeName-2"] = get_abbr_name(kgs.kg2.get_ent_name_by_id(cent_index_r))
            sub_graph_dict["centerNodeId-1"] = None
            sub_graph_dict["centerNodeId-2"] = None

            entity_set_l, triple_set_l = generate_node_and_edge_sets(kgs.kg1, cent_index_l, hop_num)
            entity_set_r, triple_set_r = generate_node_and_edge_sets(kgs.kg2, cent_index_r, hop_num)

            entity_new_id_dict = dict()
            ent_info_list_l, entity_new_id_dict = generate_node_info_list(kgs.kg1, entity_set_l, entity_new_id_dict, "KG1")
            ent_info_list_r, entity_new_id_dict = generate_node_info_list(kgs.kg2, entity_set_r, entity_new_id_dict, "KG2")

            edge_info_list_l = generate_edge_info_list(kgs.kg1, triple_set_l, entity_new_id_dict)
            edge_info_list_r = generate_edge_info_list(kgs.kg2, triple_set_r, entity_new_id_dict)

            node_list = ent_info_list_l + ent_info_list_r
            edge_list = edge_info_list_l + edge_info_list_r

            node_mappings = list()
            for item in entity_set_l:
                cp = ent_mappings_dict.get(item, None)
                if cp is not None:
                    if cp in entity_set_r:
                        mapping_info_dict = dict()
                        index = entity_new_id_dict[item]
                        cp_index = entity_new_id_dict[cp]
                        probability = ent_mappings_prob[(item, cp)]
                        mapping_info_dict["sourceNodeId"] = index
                        mapping_info_dict["targetNodeId"] = cp_index
                        mapping_info_dict["edgeName"] = "equivalent"
                        mapping_info_dict["prob"] = probability
                        node_mappings.append(mapping_info_dict)

            sub_graph_dict["nodes"] = node_list
            sub_graph_dict["edges"] = edge_list
            sub_graph_dict["mappings"] = node_mappings

            sub_graph_dict["centerNodeId-1"] = entity_new_id_dict[cent_index_l]
            sub_graph_dict["centerNodeId-2"] = entity_new_id_dict[cent_index_r]

    construction(demo_dict, hop)

    correction_list = generate_pairs_for_correction(kgs)
    demo_dict["markingList"] = correction_list
    demo_dict["info"]["markingNum"] = len(correction_list)

    with open(save_path, "w", encoding="utf8") as f:
        json.dump(demo_dict, f, indent=4)




