from utils import *
import pickle
import sys

# rank Shannon discrimination measure

g = Log()
h = Sum()

f_r = Dsr()
rsdm = Gdm(h, g, f_r)

start = time.time()

dataset, t = generate_2Ddataset(0, 2, 1000, 0, 0.1, [[-10, 10], [-10, 10]])

print("noise (rsdm): ", 0.05 * int(sys.argv[2]))

# adding noise to the dataset
for k in range(int(sys.argv[2])):
    dataset = add_noise(dataset, 0.05)

sets = get_ten_folds(dataset)

acc = 0
ratio = 0
nb_examples = 0

for i in range(10):
    test_set = sets[i]
    train_set = LabeledSet(2)
    for j in range(10):
        if i != j:
            train_set.addExamples(sets[j].x, sets[j].y)
    tree = RDMT(rsdm, "shannon", 0, 100, 0.01, [1, 2])
    tree.train(train_set)
    acc += tree.accuracy(test_set)

    #print("(rsdm) BEGIN get_ratio_non_monotone_pairs : ", time.time())
    ratio += tree.get_ratio_non_monotone_pairs()
    #print("(rsdm) END get_ratio_non_monotone_pairs : ", time.time())

    #print("(rsdm) BEGIN get_ratio_non_monotone_pairs : ", time.time())
    nb_examples += tree.get_total_examples_ratio()
    #print("(rsdm) END get_ratio_non_monotone_pairs : ", time.time())


    print("Iter {} tree (rsdm)".format(i))


acc = acc * (1.0/10)
ratio = ratio * (1.0/10)
nb_examples = nb_examples * (1.0/10)

print(acc, ratio, nb_examples)

#print("Running time (rsdm) (" + sys.argv[2]+ ") : " + str(time.time() - start))
#f_acc = open("acc1_" + sys.argv[2], "wb")
#f_leaves = open("leaves1_"+ sys.argv[2], "wb")
#f_depth = open("depth1_"+ sys.argv[2], "wb")
#f_ratio = open("ratio1_"+ sys.argv[2], "wb")
#f_pairs = open("pairs1_"+ sys.argv[2], "wb")
#
#pickle.dump(acc, f_acc)
#pickle.dump(leaves, f_leaves)
#pickle.dump(depth, f_depth)
#pickle.dump(ratio, f_ratio)
#pickle.dump(pairs, f_pairs)
#
#f_acc.close()
#f_leaves.close()
#f_depth.close()
#f_ratio.close()
#f_pairs.close()
