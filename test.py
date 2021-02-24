import KG
import KGs
import time


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


path_r_1 = r"D:\repos\PARIS-PYTHON\dataset\EN_DE_15K_V2\rel_triples_1"
path_r_2 = r"D:\repos\PARIS-PYTHON\dataset\EN_DE_15K_V2\rel_triples_2"

path_a_1 = r"D:\repos\PARIS-PYTHON\dataset\EN_DE_15K_V2\attr_triples_1"
path_a_2 = r"D:\repos\PARIS-PYTHON\dataset\EN_DE_15K_V2\attr_triples_2"

test_path = r"D:\repos\PARIS-PYTHON\dataset\EN_DE_15K_V2\ent_links"

start = time.time()

kg1 = construct_kg(path_r_1, path_a_1)

# kg1.kg_core.test()

kg2 = construct_kg(path_r_2, path_a_2)

kgs = KGs.KGs(kg1, kg2)

kgs.init()
kgs.run()

kgs.test(test_path, [0.1 * i for i in range(10)])

end = time.time()

# kgs.print_result()

print(end - start)
