import argparse

import se
import sys
import utils.PRASEUtils as pu
import utils.VLDBDemoUtils as vu

parser = argparse.ArgumentParser(description="Probabilistic Reasoning and Semantic Embedding")

parser.add_argument("--kg1_rel_path", type=str, help="the path of KG1's relation triple file")
parser.add_argument("--kg2_rel_path", type=str, help="the path of KG2's relation triple file")
parser.add_argument("--kg1_attr_path", type=str, help="the path of KG1's attribute triple file")
parser.add_argument("--kg2_attr_path", type=str, help="the path of KG2's attribute triple file")

parser.add_argument("--iterations", type=int, default=1, help="PRASE iteration number")
parser.add_argument("--load_path", type=str, help="load the PRASE model from path")
parser.add_argument("--save_path", type=str, help="save the PRASE model to path")
parser.add_argument("--save_emb", action="store_true", default=False, help="enable saving the entity embeddings of PRASE model")

# parser.add_argument("--disable_init_run", action="store_true", default=False, help="disable running pr first for unsupervised initialization")

parser.add_argument("--save_kg_demo_path", type=str, help="save the KG demonstration file to path")
parser.add_argument("--save_mapping_demo_path", type=str, help="save the entity mapping file to path")
parser.add_argument("--save_mapping_result_path", type=str, help="save the entity mapping results to path for user download")
parser.add_argument("--forced_file_path", type=str, help="the path of the file containing forced equivalent entity mappings")


def print_kgs_stat(kgs_obj):
    print(vu.get_time_str() + "Discovered entity mapping number: " + str(len(kgs_obj.get_ent_align_ids_result())))
    sys.stdout.flush()
    rel_align_a, rel_align_b = kgs_obj.get_rel_align_ids_result()
    print(vu.get_time_str() + "Discovered relation mapping number: " + str(len(rel_align_a) + len(rel_align_b)))
    sys.stdout.flush()
    attr_align_a, attr_align_b = kgs_obj.get_attr_align_name_result()
    print(vu.get_time_str() + "Discovered attribute mapping number: " + str(len(attr_align_a) + len(attr_align_b)))
    sys.stdout.flush()


def print_kg_stat(kg_obj):
    print(vu.get_time_str() + "Entity Number: " + str(len(kg_obj.get_ent_id_set())))
    print(vu.get_time_str() + "Relation Number: " + str(int(len(kg_obj.get_rel_id_set()) / 2)))
    print(vu.get_time_str() + "Attribute Number: " + str(int(len(kg_obj.get_attr_id_set()) / 2)))
    print(vu.get_time_str() + "Literal Number: " + str(int(len(kg_obj.get_lite_id_set()) / 2)))
    print(vu.get_time_str() + "Relation Triple Number: " + str(int(len(kg_obj.get_relation_id_triples()) / 2)))
    print(vu.get_time_str() + "Attribute Triple Number: " + str(int(len(kg_obj.get_attribute_id_triples()) / 2)))
    sys.stdout.flush()


if __name__ == '__main__':
    try:
        args = parser.parse_args()
        kg1_rel_path, kg2_rel_path = args.kg1_rel_path, args.kg2_rel_path
        kg1_attr_path, kg2_attr_path = args.kg1_attr_path, args.kg2_attr_path
        iteration = args.iterations
        # init_run = not args.disable_init_run

        load_path = args.load_path
        save_path = args.save_path
        save_emb = args.save_emb

        save_kg_demo_path = args.save_kg_demo_path
        save_mapping_demo_path = args.save_mapping_demo_path
        save_mapping_result_path = args.save_mapping_result_path
        forced_file_path = args.forced_file_path

        init_run = load_path is None

        print(vu.get_time_str() + "Construct source KG...")
        sys.stdout.flush()
        kg1 = pu.construct_kg(kg1_rel_path, kg1_attr_path)
        print_kg_stat(kg1)

        print(vu.get_time_str() + "Construct target KG...")
        sys.stdout.flush()
        kg2 = pu.construct_kg(kg2_rel_path, kg2_attr_path)
        print_kg_stat(kg2)

        print(vu.get_time_str() + "Configure PRASESys...")
        sys.stdout.flush()
        kgs = pu.construct_kgs(kg1, kg2, se.GCNAlign, epoch_num=100)
        kgs.pr.set_worker_num(4)
        kgs.init()

        if load_path is not None:
            print(vu.get_time_str() + "Loading PRASESys...")
            sys.stdout.flush()
            pu.load_prase_model(kgs, load_path)
            print_kgs_stat(kgs)
            kgs.pr.enable_rel_init(False)

        if forced_file_path is not None:
            print(vu.get_time_str() + "Loading user feedback...")
            sys.stdout.flush()
            vu.load_forced_ent_mappings(kgs, forced_file_path)

        if init_run:
            print(vu.get_time_str() + "Performing initial PR Module (PARIS)...")
            sys.stdout.flush()
            kgs.run_pr()
            print_kgs_stat(kgs)

        for i in range(iteration):
            print(vu.get_time_str() + "Performing SE Module (GCNAlign)...")
            sys.stdout.flush()
            kgs.run_se(embedding_feedback=True, mapping_feedback=True)
            print_kgs_stat(kgs)
            print(vu.get_time_str() + "Performing PR Module (PARIS)...")
            sys.stdout.flush()
            kgs.run_pr()
            print_kgs_stat(kgs)

        if save_path is not None:
            pu.save_prase_model(kgs, save_path, save_emb=save_emb)

        if save_kg_demo_path is not None:
            vu.construct_single_kg_demo_file(kgs, save_kg_demo_path)

        if save_mapping_demo_path is not None:
            vu.construct_kg_mappings_demo_file(kgs, save_mapping_demo_path)

        if save_mapping_result_path is not None:
            vu.save_ent_mapping_result(kgs, save_mapping_result_path)

        print(vu.get_time_str() + "Task accomplished")

    except BaseException:
        print(vu.get_time_str() + "Task failed")
