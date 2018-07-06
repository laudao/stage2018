from utils import *
import pickle
import sys

# Gini entropy

g = One_minus()
h = Sum()

f = Ds()
gdm = Gdm(h, g, f)

start = time.time()

dataset, t = generate_2Ddataset(0, 3, 1000, 0, 0.1, [[-10, 10], [-10, 10]])

print("noise (gdm): ", 0.05 * int(sys.argv[2]))

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

for i in range(10):
    test_set = sets[i]
    train_set = LabeledSet(2)
    for j in range(10):
        if i != j:
            train_set.addExamples(sets[j].x, sets[j].y)
    tree = RDMT(gdm, "shannon", 0, 100, 0.01, [1, 2, 3])
    tree.train(train_set)
    acc += tree.accuracy(test_set)

    print("(gdm) BEGIN get_nb_leaves : ", time.time())
    leaves += tree.get_nb_leaves()
    print("(gdm) END get_nb_leaves : ", time.time())

    print("(gdm) BEGIN get_depth : ", time.time())
    depth += tree.get_depth()
    print("(gdm) END get_depth : ", time.time())

    print("(gdm) BEGIN get_ratio_non_monotone_pairs : ", time.time())
    ratio += tree.get_ratio_non_monotone_pairs()
    print("(gdm) END get_ratio_non_monotone_pairs : ", time.time())

    print("(gdm) BEGIN get_total_pairs : ", time.time())
    pairs += tree.get_total_pairs()
    print("(gdm) END get_total_pairs : ", time.time())

    print("(gdm) BEGIN evaluate_monotonicity : ", time.time())
    evaluation += tree.evaluate_monotonicity()
    print("(gdm) END evaluate_monotonicity : ", time.time())

    print("(gdm) BEGIN pairs_ratio : ", time.time())
    p_ratio += tree.pairs_ratio()
    print("(gdm) END pairs_ratio : ", time.time())

    print("Iter {} tree (gdm)".format(i))


acc = acc * (1.0/10)
depth = depth * (1.0/10)
ratio = ratio * (1.0/10)
pairs = pairs * (1.0/10)
evaluation = evaluation * (1.0/10)
p_ratio = p_ratio * (1.0/10)


print("Running time (gdm) (" + sys.argv[2]+ ") : " + str(time.time() - start))
f_acc = open("acc3_4_" + sys.argv[2], "wb")
f_leaves = open("leaves3_4_"+ sys.argv[2], "wb")
f_depth = open("depth3_4_"+ sys.argv[2], "wb")
f_ratio = open("ratio3_4_"+ sys.argv[2], "wb")
f_pairs = open("pairs3_4_"+ sys.argv[2], "wb")
f_eval = open("eval3_4_"+ sys.argv[2], "wb")
f_pratio = open("p_ratio3_4_"+sys.argv[2], "wb")

pickle.dump(acc, f_acc)
pickle.dump(leaves, f_leaves)
pickle.dump(depth, f_depth)
pickle.dump(ratio, f_ratio)
pickle.dump(pairs, f_pairs)
pickle.dump(evaluation, f_eval)
pickle.dump(p_ratio, f_pratio)

f_acc.close()
f_leaves.close()
f_depth.close()
f_ratio.close()
f_pairs.close()
f_eval.close()
f_pratio.close()
