from utils import *
import pickle
import sys

# H_M
g = One_minus_square()
h = Max()
f = Dsr()
hm = Gdm(h, g, f)

start = time.time()

dataset, t = generate_2Ddataset(0, 3, 1000, 0, 0.1, [[-10, 10], [-10, 10]])

print("noise (hm): ", 0.05 * int(sys.argv[2]))

# adding noise to the dataset
for k in range(int(sys.argv[2])):
    dataset = add_noise(dataset, 0.05)

sets = get_ten_folds(dataset)

acc = 0
leaves = 0
depth = 0
ratio = 0
pairs = 0
evaluation = 0
p_ratio = 0
nb_examples = 0

for i in range(10):
    test_set = sets[i]
    train_set = LabeledSet(2)
    for j in range(10):
        if i != j:
            train_set.addExamples(sets[j].x, sets[j].y)
    tree = RDMT(hm, "shannon", 0, 100, 0.01, [1, 2, 3])
    tree.train(train_set)
    acc += tree.accuracy(test_set)

    print("(hm) BEGIN get_nb_leaves : ", time.time())
    leaves += tree.get_nb_leaves()
    print("(hm) END get_nb_leaves : ", time.time())

    print("(hm) BEGIN get_depth : ", time.time())
    depth += tree.get_depth()
    print("(hm) END get_depth : ", time.time())

    print("(hm) BEGIN get_ratio_non_monotone_pairs : ", time.time())
    ratio += tree.get_ratio_non_monotone_pairs()
    print("(hm) END get_ratio_non_monotone_pairs : ", time.time())

    print("(hm) BEGIN get_total_pairs : ", time.time())
    pairs += tree.get_total_pairs()
    print("(hm) END get_total_pairs : ", time.time())

    print("(hm) BEGIN evaluate_monotonicity : ", time.time())
    evaluation += tree.evaluate_monotonicity()
    print("(hm) END evaluate_monotonicity : ", time.time())

    print("(hm) BEGIN pairs_ratio : ", time.time())
    p_ratio += tree.pairs_ratio()
    print("(hm) END pairs_ratio : ", time.time())

    print("(hm) BEGIN get_total_examples_ratio : ", time.time())
    nb_examples += tree.get_total_examples_ratio()
    print("(hm) END get_total_examples_ratio : ", time.time())

    print("Iter {} tree (hm)".format(i))


acc = acc * (1.0/10)
depth = depth * (1.0/10)
ratio = ratio * (1.0/10)
pairs = pairs * (1.0/10)
evaluation = evaluation * (1.0/10)
p_ratio = p_ratio * (1.0/10)
nb_examples = nb_examples * (1.0/10)

print("Running time (hm) (" + sys.argv[2]+ ") : " + str(time.time() - start))
f_acc = open("acc3_6_" + sys.argv[2], "wb")
f_leaves = open("leaves3_6_"+ sys.argv[2], "wb")
f_depth = open("depth3_6_"+ sys.argv[2], "wb")
f_ratio = open("ratio3_6_"+ sys.argv[2], "wb")
f_pairs = open("pairs3_6_"+ sys.argv[2], "wb")
f_eval = open("eval3_6_"+ sys.argv[2], "wb")
f_pratio = open("p_ratio3_6_"+sys.argv[2], "wb")
f_nb_examples = open("nb_examples3_6_" + sys.argv[2], "wb")

pickle.dump(acc, f_acc)
pickle.dump(leaves, f_leaves)
pickle.dump(depth, f_depth)
pickle.dump(ratio, f_ratio)
pickle.dump(pairs, f_pairs)
pickle.dump(evaluation, f_eval)
pickle.dump(p_ratio, f_pratio)
pickle.dump(nb_examples, f_nb_examples)

f_acc.close()
f_leaves.close()
f_depth.close()
f_ratio.close()
f_pairs.close()
f_eval.close()
f_pratio.close()
f_nb_examples.close()
