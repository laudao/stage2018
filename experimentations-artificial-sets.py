from utils import *
import pickle
import sys

g = Log()
h = Sum()

# rank Shannon discrimination measure
f_r = Dsr()
rsdm = Gdm(h, g, f_r)

# conditional Shannon entropy
f = Ds()
sdm = Gdm(h, g, f)

# pessimistic rank discrimination measure
f_p = Mindsr()
g_p = Frac()
prdm = Gdm(h, g_p, f_p)

start = time.time()

dataset, t = generate_2Ddataset(0, 2, 1000, 0, 0.1, [[-10, 10], [-10, 10]])

print("noise : ", 0.05 * int(sys.argv[2]))

for k in range(int(sys.argv[2])):
    dataset = add_noise(dataset, 0.05)

sets = get_ten_folds(dataset)

acc1 = 0
leaves1 = 0
depth1 = 0
ratio1 = 0
pairs1 = 0

acc2 = 0
leaves2 = 0
depth2 = 0
ratio2 = 0
pairs2 = 0

acc3 = 0
leaves3 = 0
depth3 = 0
ratio3 = 0
pairs3 = 0

for i in range(10):
    test_set = sets[i]
    train_set = LabeledSet(2)
    for j in range(10):
        if i != j:
            train_set.addExamples(sets[j].x, sets[j].y)
    tree1 = RDMT(rsdm, "shannon", 0, 100, 0.01, [1, 2])
    tree1.train(train_set)
    acc1 += tree1.accuracy(test_set)
    leaves1 += tree1.get_nb_leaves()
    depth1 += tree1.get_depth()
    ratio1 += tree1.get_ratio_non_monotone_pairs()
    pairs1 += tree1.get_total_pairs()
    print("Iter {} tree1".format(i))

    tree2 = RDMT(sdm, "shannon", 0, 100, 0.01, [1, 2])
    tree2.train(train_set)
    acc2 += tree2.accuracy(test_set)
    leaves2 += tree2.get_nb_leaves()
    depth2 += tree2.get_depth()
    ratio2 += tree2.get_ratio_non_monotone_pairs()
    pairs2 += tree2.get_total_pairs()

    print("Iter {} tree2".format(i))
    tree3 = RDMT(prdm, "shannon", 0, 100, 0.01, [1, 2])
    tree3.train(train_set)
    acc3 += tree3.accuracy(test_set)
    leaves3 += tree3.get_nb_leaves()
    depth3 += tree3.get_depth()
    ratio3 += tree3.get_ratio_non_monotone_pairs()
    pairs3 += tree3.get_total_pairs()
    print("Iter {} tree3".format(i))

acc1 = acc1 * (1.0/10)
depth1 = depth1 * (1.0/10)
ratio1 = ratio1 * (1.0/10)
pairs1 = pairs1 * (1.0/10)

acc2 = acc2 * (1.0/10)
leaves2 = leaves2 * (1.0/10)
depth2 = depth2 * (1.0/10)
ratio2 = ratio2 * (1.0/10)
pairs2 = pairs2 * (1.0/10)

acc3 = acc3 * (1.0/10)
leaves3 = leaves3 * (1.0/10)
depth3 = depth3 * (1.0/10)
ratio3 = ratio3 * (1.0/10)
pairs3 = pairs3 * (1.0/10)

print("Running time (" + sys.argv[2]+ ") : " + str(time.time() - start))
f_acc1 = open("acc1_" + sys.argv[2], "wb")
f_leaves1 = open("leaves1_"+ sys.argv[2], "wb")
f_depth1 = open("depth1_"+ sys.argv[2], "wb")
f_ratio1 = open("ratio1_"+ sys.argv[2], "wb")
f_pairs1 = open("pairs1_"+ sys.argv[2], "wb")


f_acc2 = open("acc2_"+ sys.argv[2], "wb")
f_leaves2 = open("leaves2_"+ sys.argv[2], "wb")
f_depth2 = open("depth2_"+ sys.argv[2], "wb")
f_ratio2 = open("ratio2_"+ sys.argv[2], "wb")
f_pairs2 = open("pairs2_"+ sys.argv[2], "wb")

f_acc3 = open("acc3_"+ sys.argv[2], "wb")
f_leaves3 = open("leaves3_"+ sys.argv[2], "wb")
f_depth3 = open("depth3_"+ sys.argv[2], "wb")
f_ratio3 = open("ratio3_"+ sys.argv[2], "wb")
f_pairs3 = open("pairs3_"+ sys.argv[2], "wb")

pickle.dump(acc1, f_acc1)
pickle.dump(leaves1, f_leaves1)
pickle.dump(depth1, f_depth1)
pickle.dump(ratio1, f_ratio1)
pickle.dump(pairs1, f_pairs1)

pickle.dump(acc2, f_acc2)
pickle.dump(leaves2, f_leaves2)
pickle.dump(depth2, f_depth2)
pickle.dump(ratio2, f_ratio2)
pickle.dump(pairs2, f_pairs2)

pickle.dump(acc3, f_acc3)
pickle.dump(leaves3, f_leaves3)
pickle.dump(depth3, f_depth3)
pickle.dump(ratio3, f_ratio3)
pickle.dump(pairs3, f_pairs3)


f_acc1.close()
f_leaves1.close()
f_depth1.close()
f_ratio1.close()
f_pairs1.close()


f_acc2.close()
f_leaves2.close()
f_depth2.close()
f_ratio2.close()
f_pairs2.close()

f_acc3.close()
f_leaves3.close()
f_depth3.close()
f_ratio3.close()
f_pairs3.close()
