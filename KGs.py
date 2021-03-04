import KG
import prase_core as pc


class KGs:
    default_lite_align_prob = 0.99

    def __init__(self, kg1: KG, kg2: KG):
        self.kg1 = kg1
        self.kg2 = kg2

        self.pr = pc.PRModule(kg1.kg_core, kg2.kg_core)

    def align_literals(self):
        for (lite_id, lite_name) in self.kg1.lite_id_name_dict.items():
            if self.kg2.name_lite_id_dict.__contains__(lite_name):
                lite_cp_id = self.kg2.name_lite_id_dict[lite_name]
                self.pr.insert_lite_eqv(lite_id, lite_cp_id, KGs.default_lite_align_prob, False)
                self.pr.insert_lite_eqv(lite_cp_id, lite_id, KGs.default_lite_align_prob, False)

    def init(self):
        self.pr.init()

    def run_pr(self):
        self.pr.run()

    def get_entity_nums(self):
        return len(self.kg1.get_ent_id_set()) + len(self.kg2.get_ent_id_set())

    def get_attribute_nums(self):
        return len(self.kg1.get_a()) + len(self.kg2.get_ent_id_set())

    def insert_forced_ent_eqv(self, ent_name, ent_cp_name, prob):
        ent_id = self.kg1.get_ent_id_without_insert(ent_name)
        ent_cp_id = self.kg2.get_ent_id_without_insert(ent_cp_name)
        if ent_id is None or ent_cp_id is None:
            print("fail to get ent ids")
            return
        self.pr.insert_ent_eqv(ent_id, ent_cp_id, prob, True)
        self.pr.insert_ent_eqv(ent_cp_id, ent_id, prob, True)

    def insert_ent_eqv(self, ent_name, ent_cp_name, prob):
        ent_id = self.kg1.get_ent_id_without_insert(ent_name)
        ent_cp_id = self.kg2.get_ent_id_without_insert(ent_cp_name)
        if ent_id is None or ent_cp_id is None:
            print("fail to get ent ids")
            return
        self.pr.insert_ent_eqv(ent_id, ent_cp_id, prob, False)
        self.pr.insert_ent_eqv(ent_cp_id, ent_id, prob, False)

    def insert_ent_eqv_by_id(self, ent_id, ent_cp_id, prob):
        if ent_id is None or ent_cp_id is None:
            print("fail to get ent ids")
            return
        self.pr.insert_ent_eqv(ent_id, ent_cp_id, prob, False)
        self.pr.insert_ent_eqv(ent_cp_id, ent_id, prob, False)

    def get_kg1_unaligned_candidates(self):
        return self.pr.get_kg_a_unaligned_ents()

    def get_kg2_unaligned_candidates(self):
        return self.pr.get_kg_b_unaligned_ents()

    def clear_kgs_ent_embed(self):
        self.kg1.clear_ent_embed()
        self.kg2.clear_ent_embed()
        self.pr.reset_emb_eqv()

    def get_ent_align_result(self):
        return self.pr.get_ent_eqv_result()

    def print_result(self):
        for item in self.pr.get_ent_eqv_result():
            print(item)

    def test(self, test_path, threshold=0.0):
        gold_result = set()
        with open(test_path, "r", encoding="utf8") as f:
            for line in f.readlines():
                params = str.strip(line).split("\t")
                ent_l, ent_r = params[0].strip(), params[1].strip()
                obj_l, obj_r = self.kg1.get_ent_id_without_insert(ent_l), self.kg2.get_ent_id_without_insert(ent_r)
                if obj_l is None:
                    print("Exception: fail to load Entity (" + ent_l + ")")
                if obj_r is None:
                    print("Exception: fail to load Entity (" + ent_r + ")")
                if obj_l is None or obj_r is None:
                    continue
                gold_result.add((obj_l, obj_r))

        threshold_list = []
        if isinstance(threshold, float) or isinstance(threshold, int):
            threshold_list.append(float(threshold))
        else:
            threshold_list = threshold

        for threshold_item in threshold_list:
            ent_align_result = set()
            for (ent_id, counterpart_id, prob) in self.pr.get_ent_eqv_result():
                if prob > threshold_item:
                    ent_align_result.add((ent_id, counterpart_id))

            correct_num = len(gold_result & ent_align_result)
            predict_num = len(ent_align_result)
            total_num = len(gold_result)

            if predict_num == 0:
                print("Threshold: " + format(threshold_item, ".3f") + "\tException: no satisfied alignment result")
                continue

            if total_num == 0:
                print("Threshold: " + format(threshold_item, ".3f") + "\tException: no satisfied instance for testing")
            else:
                precision, recall = correct_num / predict_num, correct_num / total_num
                if precision <= 0.0 or recall <= 0.0:
                    print("Threshold: " + format(threshold_item, ".3f") + "\tPrecision: " + format(precision, ".6f") +
                          "\tRecall: " + format(recall, ".6f") + "\tF1-Score: Nan")
                else:
                    f1_score = 2.0 * precision * recall / (precision + recall)
                    print("Threshold: " + format(threshold_item, ".3f") + "\tPrecision: " + format(precision, ".6f") +
                          "\tRecall: " + format(recall, ".6f") + "\tF1-Score: " + format(f1_score, ".6f"))
