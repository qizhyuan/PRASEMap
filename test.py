import KG
import KGs
import time
from se.GCNAlign.Model import GCNAlign
import numpy as np


def construct_kg(path_r, path_a=None, sep='\t'):
    kg = KG.KG()

    with open(path_r, "r", encoding="utf-8") as f:
        for line in f.readlines():
            if len(line.strip()) == 0:
                continue
            params = str.strip(line).split(sep=sep)
            if len(params) != 3:
                print(line)
                continue
            h, r, t = params[0].strip(), params[1].strip(), params[2].strip()
            kg.insert_rel_triple(h, r, t)

    with open(path_a, "r", encoding="utf-8") as f:
        for line in f.readlines():
            if len(line.strip()) == 0:
                continue
            params = str.strip(line).split(sep=sep)
            if len(params) != 3:
                print(line)
                continue
            # assert len(params) == 3
            e, a, v = params[0].strip(), params[1].strip(), params[2].strip()
            kg.insert_attr_triple(e, a, v)

    return kg


def load_embeds(kg: KG, ent_emb_path, kg_ent_mappings):
    ent_emb = np.load(ent_emb_path)
    with open(kg_ent_mappings, "r", encoding="utf8") as f:
        for line in f.readlines():
            if len(line.strip()) == 0:
                continue
            params = line.strip().split("\t")
            ent_name, idx = params[0].strip(), int(params[1].strip())
            kg.insert_ent_embed_by_name(ent_name, ent_emb[idx, :])


path_r_1 = r"D:\repos\self\PARIS-PYTHON\dataset\industry\rel_triples_1"
path_r_2 = r"D:\repos\self\PARIS-PYTHON\dataset\industry\rel_triples_2"

path_a_1 = r"D:\repos\self\PARIS-PYTHON\dataset\industry\attr_triples_1"
path_a_2 = r"D:\repos\self\PARIS-PYTHON\dataset\industry\attr_triples_2"

test_path = r"D:\repos\self\PARIS-PYTHON\dataset\industry\ent_links"

ent_emb_path = r"D:\repos\self\PARIS-PYTHON\output\industry\GCNAlign\ent_embeds.npy"
ent_a_mapping_path = r"D:\repos\self\PARIS-PYTHON\output\industry\GCNAlign\kg1_ent_ids"
ent_b_mapping_path = r"D:\repos\self\PARIS-PYTHON\output\industry\GCNAlign\kg2_ent_ids"

start = time.time()

kg1 = construct_kg(path_r_1, path_a_1)

kg2 = construct_kg(path_r_2, path_a_2)

kgs = KGs.KGs(kg1, kg2)
kgs.align_literals()

kgs.init()
# kgs.pr.set_worker_num(6)
kgs.pr.set_se_trade_off(0.2)
kgs.run_pr()

kgs.test(test_path, [0.1 * i for i in range(10)])
gcn = GCNAlign(kgs)

for i in range(10):
    gcn.init()
    gcn.train_embeddings()
    gcn.feed_back_to_pr_module()
    kgs.run_pr()
    kgs.test(test_path, [0.1 * i for i in range(10)])

end = time.time()

print(end - start)
