import argparse

import utils.PRASEUtils as pu
import utils.VLDBDemoUtils as vu
import se

parser = argparse.ArgumentParser(description="Probabilistic Reasoning and Semantic Embedding")

parser.add_argument("--kg1_rel_path", type=str, help="the path of KG1's relation triple file")
parser.add_argument("--kg2_rel_path", type=str, help="the path of KG2's relation triple file")
parser.add_argument("--kg1_attr_path", type=str, help="the path of KG1's attribute triple file")
parser.add_argument("--kg2_attr_path", type=str, help="the path of KG2's attribute triple file")

parser.add_argument("--ent_mapping_path", type=str, help="the path of the file containing equivalent entity mappings")

parser.add_argument("--iterations", type=int, default=1, help="PRASE iteration number")
parser.add_argument("--load_path", type=str, help="load the PRASE model from path")
parser.add_argument("--save_path", type=str, help="save the PRASE model to path")
parser.add_argument("--save_emb", action="store_true", default=False, help="enable saving the entity embeddings of PRASE model")

parser.add_argument("--disable_init_run", action="store_true", default=False, help="disable running pr first for unsupervised initialization")

parser.add_argument("--save_kg_demo_path", type=str, help="save the KG demonstration file to path")
parser.add_argument("--save_mapping_demo_path", type=str, help="save the entity mapping file to path")

if __name__ == '__main__':
    args = parser.parse_args()
    kg1_rel_path, kg2_rel_path = args.kg1_rel_path, args.kg2_rel_path
    kg1_attr_path, kg2_attr_path = args.kg1_attr_path, args.kg2_attr_path
    iteration = args.iterations
    init_run = not args.disable_init_run

    load_path = args.load_path
    save_path = args.save_path
    save_emb = args.save_emb

    save_kg_demo_path = args.save_kg_demo_path
    save_mapping_demo_path = args.save_mapping_demo_path

    kg1 = pu.construct_kg(kg1_rel_path, kg1_attr_path)
    kg2 = pu.construct_kg(kg2_rel_path, kg2_attr_path)

    kgs = pu.construct_kgs(kg1, kg2, se.GCNAlign)
    kgs.init()

    # kgs.test(test_path=r"D:\repos\self\PARIS-PYTHON\dataset\industry\ent_links")
    if load_path is not None:
        pu.load_prase_model(kgs, load_path)
        kgs.pr.enable_rel_init(False)

    # kgs.test(test_path=r"D:\repos\self\PARIS-PYTHON\dataset\industry\ent_links", threshold=[0.1 * i for i in range(10)])

    if init_run:
        kgs.run_pr()

    kgs.test(test_path=r"D:\repos\self\PARIS-PYTHON\dataset\EN_DE_15K_V2\ent_links", threshold=[0.1 * i for i in range(10)])
    for i in range(iteration):
        kgs.run_se()
        kgs.run_pr()
        kgs.test(test_path=r"D:\repos\self\PARIS-PYTHON\dataset\EN_DE_15K_V2\ent_links",
                 threshold=[0.1 * i for i in range(10)])

    if save_path is not None:
        pu.save_prase_model(kgs, save_path, save_emb=save_emb)

    if save_kg_demo_path is not None:
        vu.construct_single_kg_demo_file(kgs, save_kg_demo_path)

    if save_mapping_demo_path is not None:
        vu.construct_kg_mappings_demo_file(kgs, save_mapping_demo_path)
