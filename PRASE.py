import argparse

import utils.PRASEUtils as pu
from se.GCNAlign.Model import GCNAlign

parser = argparse.ArgumentParser(description="Probabilistic Reasoning and Semantic Embedding")

parser.add_argument("--kg1_rel_path", type=str, help="the path of KG1's relation triple file")
parser.add_argument("--kg2_rel_path", type=str, help="the path of KG2's relation triple file")
parser.add_argument("--kg1_attr_path", type=str, help="the path of KG1's attribute triple file")
parser.add_argument("--kg2_attr_path", type=str, help="the path of KG2's attribute triple file")

parser.add_argument("--ent_mapping_path", type=str, help="the path of the file containing equivalent entity mappings")

parser.add_argument("--iterations", type=int, default=1, help="PRASE iteration number")
parser.add_argument("--load_path", type=str, help="load the PRASE model from path")
parser.add_argument("--save_path", type=str, help="save the PRASE model from path")

parser.add_argument("--init_run", type=bool, default=True, help="run pr first for unsupervised initialization")


if __name__ == '__main__':
    args = parser.parse_args()
    kg1_rel_path, kg2_rel_path = args.kg1_rel_path, args.kg2_rel_path
    kg1_attr_path, kg2_attr_path = args.kg1_attr_path, args.kg2_attr_path
    iteration = args.iterations
    init_run = args.init_run
    load_path = args.load_path

    kg1 = pu.construct_kg(kg1_rel_path, kg1_attr_path)
    kg2 = pu.construct_kg(kg2_rel_path, kg2_attr_path)

    kgs = pu.construct_kgs(kg1, kg2, GCNAlign)
    kgs.init()

    if load_path is not None:
        pu.load_prase_model(kgs, load_path)

    kgs.test(test_path=r"D:\repos\self\PARIS-PYTHON\dataset\EN_DE_15K_V2\ent_links")

    # if init_run:
    #     kgs.run_pr()
    # for i in range(iteration):
    #     kgs.run_se()
    #     kgs.run_pr()

    # kgs.test(test_path=r"D:\repos\self\PARIS-PYTHON\dataset\industry\ent_links")
    # print(kgs.get_ent_align_name_result())
    pu.save_prase_model(kgs, "./model.json")
