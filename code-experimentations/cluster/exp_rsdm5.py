from utils import *
import pickle
import sys

# rank Shannon discrimination measure

g = Log()
h = Sum()

f = Dsr()
rsdm = Gdm(h, g, f)

start = time.time()

dataset = generate_monotone_consistent_dataset(500, 5)

print("noise (rsdm): ", 0.05 * int(sys.argv[2]))

# adding noise to the dataset
for k in range(int(sys.argv[2])):
    dataset = add_noise(dataset, 0.05)

sets = get_ten_folds(dataset)

acc = 0
leaves = 0
depth = 0
ratio = 0
nb_examples = 0
pairs = 0
evaluation = 0

for i in range(10):
    test_set = sets[i]
    train_set = LabeledSet(2)
    for j in range(10):
        if i != j:
            train_set.addExamples(sets[j].x, sets[j].y)
    tree = RDMT(rsdm, "shannon", 0, 100, 0.1 * train_set.size(), [1, 2, 3, 4, 5])
    tree.train(train_set)
    acc += tree.accuracy(test_set)

    print("(rsdm) BEGIN get_nb_leaves : ", time.time())
    leaves += tree.get_nb_leaves()
    print("(rsdm) END get_nb_leaves : ", time.time())

    print("(rsdm) BEGIN get_depth : ", time.time())
    depth += tree.get_depth()
    print("(rsdm) END get_depth : ", time.time())

    print("(rsdm) BEGIN get_ratio_non_monotone_pairs : ", time.time())
    ratio += tree.get_ratio_non_monotone_pairs()
    print("(rsdm) END get_ratio_non_monotone_pairs : ", time.time())

    print("(rsdm) BEGIN avg_nb_examples_per_pair : ", time.time())
    nb_examples += tree.avg_nb_examples_per_pair()
    print("(rsdm) END avg_nb_examples_per_pair : ", time.time())

    print("(rsdm) BEGIN get_total_pairs : ", time.time())
    pairs += tree.get_total_pairs()
    print("(rsdm) END get_total_pairs : ", time.time())

    print("(rsdm) BEGIN evaluate_monotonicity : ", time.time())
    evaluation += tree.evaluate_monotonicity()
    print("(rsdm) END evaluate_monotonicity : ", time.time())

    print("Iter {} tree (rsdm)".format(i))


acc = acc * (1.0/10)
leaves = leaves * (1.0/10)
depth = depth * (1.0/10)
ratio = ratio * (1.0/10)
nb_examples = nb_examples * (1.0/10)
pairs = pairs * (1.0/10)
evaluation = evaluation * (1.0/10)

print("Running time (rsdm) (" + sys.argv[2]+ ") : " + str(time.time() - start))
f_acc = open("k5acc1_" + sys.argv[2], "wb")
f_leaves = open("k5leaves1_"+ sys.argv[2], "wb")
f_depth = open("k5depth1_"+ sys.argv[2], "wb")
f_ratio = open("k5ratio1_"+ sys.argv[2], "wb")
f_examples = open("k5examples1_"+ sys.argv[2], "wb")
f_pairs = open("k5pairs1_"+ sys.argv[2], "wb")
f_eval = open("k5eval1_" + sys.argv[2], "wb")

pickle.dump(acc, f_acc)
pickle.dump(leaves, f_leaves)
pickle.dump(depth, f_depth)
pickle.dump(ratio, f_ratio)
pickle.dump(nb_examples, f_examples)
pickle.dump(pairs, f_pairs)
pickle.dump(evaluation, f_eval)

f_acc.close()
f_leaves.close()
f_depth.close()
f_ratio.close()
f_examples.close()
f_pairs.close()
f_eval.close()
