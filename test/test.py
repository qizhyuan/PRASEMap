import os
import sys
import pr
import se
import utils.PRASEUtils as pu
from time import strftime, localtime


def get_time_str():
    return str(strftime("[%Y-%m-%d %H:%M:%S]: ", localtime()))


def print_kgs_stat(kgs_obj):
    print(get_time_str() + "Discovered entity mapping number: " + str(len(kgs_obj.get_ent_align_ids_result())))
    sys.stdout.flush()
    rel_align_a, rel_align_b = kgs_obj.get_rel_align_ids_result()
    print(get_time_str() + "Discovered relation mapping number: " + str(len(rel_align_a) + len(rel_align_b)))
    sys.stdout.flush()
    attr_align_a, attr_align_b = kgs_obj.get_attr_align_name_result()
    print(get_time_str() + "Discovered attribute mapping number: " + str(len(attr_align_a) + len(attr_align_b)))
    sys.stdout.flush()


def print_kg_stat(kg_obj):
    print(get_time_str() + "Entity Number: " + str(len(kg_obj.get_ent_id_set())))
    print(get_time_str() + "Relation Number: " + str(int(len(kg_obj.get_rel_id_set()) / 2)))
    print(get_time_str() + "Attribute Number: " + str(int(len(kg_obj.get_attr_id_set()) / 2)))
    print(get_time_str() + "Literal Number: " + str(int(len(kg_obj.get_lite_id_set()) / 2)))
    print(get_time_str() + "Relation Triple Number: " + str(int(len(kg_obj.get_relation_id_triples()) / 2)))
    print(get_time_str() + "Attribute Triple Number: " + str(int(len(kg_obj.get_attribute_id_triples()) / 2)))
    sys.stdout.flush()


path = os.path.abspath(__file__)
base, _ = os.path.split(path)

kg1_rel_path = os.path.join(base, "data/MED-BBK-9K/rel_triples_1")
kg1_attr_path = os.path.join(base, "data/MED-BBK-9K/attr_triples_1")

kg2_rel_path = os.path.join(base, "data/MED-BBK-9K/rel_triples_2")
kg2_attr_path = os.path.join(base, "data/MED-BBK-9K/attr_triples_2")

test_path = os.path.join(base, "data/MED-BBK-9K/ent_links")

print(get_time_str() + "Construct source KG...")
sys.stdout.flush()

# construct source KG from file
kg1 = pu.construct_kg(kg1_rel_path, kg1_attr_path)
print_kg_stat(kg1)

print(get_time_str() + "Construct target KG...")
sys.stdout.flush()

# construct target KG from file
kg2 = pu.construct_kg(kg2_rel_path, kg2_attr_path)
print_kg_stat(kg2)

# construct KGs object
kgs = pu.construct_kgs(kg1, kg2)

# configure kgs
kgs.set_se_module(se.GCNAlign)
kgs.set_pr_module(pr.PARIS)

'''
Set Thread Number:
kgs.pr.set_worker_num(4)
'''

'''
Load PRASEMap Model:
pu.load_prase_model(kgs, load_path)
'''

# init kgs
kgs.init()

print(get_time_str() + "Performing PR Module (PARIS)...")
sys.stdout.flush()

# run pr module
kgs.run_pr()
print_kgs_stat(kgs)
kgs.test(test_path, threshold=[0.1 * i for i in range(10)])

kgs.pr.enable_rel_init(False)

iteration = 1
for i in range(iteration):
    print(get_time_str() + "Performing SE Module (GCNAlign)...")
    sys.stdout.flush()
    # run se module
    kgs.run_se(embedding_feedback=True, mapping_feedback=True)

    print_kgs_stat(kgs)
    kgs.test(test_path, threshold=[0.1 * i for i in range(10)])
    print(get_time_str() + "Performing PR Module (PARIS)...")
    sys.stdout.flush()
    # run pr module
    kgs.run_pr()

    print_kgs_stat(kgs)
    kgs.test(test_path, threshold=[0.1 * i for i in range(10)])

'''
Save PRASEMap Model:
pu.save_prase_model(kgs, save_path)
'''
