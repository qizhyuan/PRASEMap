import prase_core as pc


class PARIS:
    def __init__(self, kgs):
        self._pr = pc.PRModule(kgs.kg1.kg_core, kgs.kg2.kg_core)

    def init(self):
        self._pr.init()

    def run(self):
        self._pr.run()

    def init_loaded_data(self):
        self._pr.init_loaded_data()

    def update_lite_eqv(self, lite_id, lite_cp_id, prob, force):
        self._pr.update_lite_eqv(lite_id, lite_cp_id, prob, force)

    def update_ent_eqv(self, ent_id, ent_cp_id, prob, force):
        self._pr.update_ent_eqv(ent_id, ent_cp_id, prob, force)

    def update_rel_eqv(self, rel_id, rel_cp_id, prob, force):
        self._pr.update_rel_eqv(rel_id, rel_cp_id, prob, force)

    def remove_forced_eqv(self, idx_a, idx_b):
        return self._pr.remove_forced_eqv(idx_a, idx_b)

    def get_kg_a_unaligned_ents(self):
        return self._pr.get_kg_a_unaligned_ents()

    def get_kg_b_unaligned_ents(self):
        return self._pr.get_kg_b_unaligned_ents()

    def reset_emb_eqv(self):
        self._pr.reset_emb_eqv()

    def get_ent_eqv_result(self):
        return self._pr.get_ent_eqv_result()

    def get_rel_eqv_result(self):
        return self._pr.get_rel_eqv_result()

    def get_forced_eqv_result(self):
        return self._pr.get_forced_eqv_result()

    def set_se_trade_off(self, tradeoff):
        self._pr.set_se_trade_off(tradeoff)

    def set_ent_candidate_num(self, candidate_num):
        self._pr.set_ent_candidate_num(candidate_num)

    def set_rel_func_bar(self, bar):
        self._pr.set_rel_func_bar(bar)

    def set_worker_num(self, workers):
        self._pr.set_worker_num(workers)

    def enable_rel_init(self, init):
        self._pr.enable_rel_init(init)
