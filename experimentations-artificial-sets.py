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

y1 = []
y2 = []
y3 = []

avg_leaves1 = []
avg_leaves2 = []
avg_leaves3 = []

avg_depth1 = []
avg_depth2 = []
avg_depth3 = []

avg_ratio1 = []
avg_ratio2 = []
avg_ratio3 = []

avg_pairs1 = []
avg_pairs2 = []
avg_pairs3 = []

start = time.time()

dataset, t = generate_2Ddataset(0, 2, 1000, 0, 0.1, [[-10, 10], [-10, 10]])
noises = []
noise = 0

for k in range(10):
    if k > 0:
        dataset = add_noise(dataset, 0.05)
    sets = get_ten_folds(dataset)
    noises.append(noise)
    noise += 0.05

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
        for j in range(0, 10):
            if i != j:
                train_set.addExamples(sets[j].x, sets[j].y)
        tree1 = RDMT(rsdm, "shannon", 0, 100, 0.01, [1, 2])
        tree1.train(train_set)
        acc1 += tree1.accuracy(test_set)
        leaves1 += tree1.get_nb_leaves()
        depth1 += tree1.get_depth()
        ratio1 += tree1.get_ratio_non_monotone_pairs()
        pairs1 += tree1.get_total_pairs()

        tree2 = RDMT(sdm, "shannon", 0, 100, 0.01, [1, 2])
        tree2.train(train_set)
        acc2 += tree2.accuracy(test_set)
        leaves2 += tree2.get_nb_leaves()
        depth2 += tree2.get_depth()
        ratio2 += tree2.get_ratio_non_monotone_pairs()
        pairs2 += tree2.get_total_pairs()

        tree3 = RDMT(prdm, "shannon", 0, 100, 0.01, [1, 2])
        tree3.train(train_set)
        acc3 += tree3.accuracy(test_set)
        leaves3 += tree3.get_nb_leaves()
        depth3 += tree3.get_depth()
        ratio3 += tree3.get_ratio_non_monotone_pairs()
        pairs3 += tree3.get_total_pairs()

    y1.append(acc1 * (1.0/10))
    avg_leaves1.append(leaves1 * (1.0/10))
    avg_depth1.append(depth1 * (1.0/10))
    avg_ratio1.append(ratio1 * (1.0/10))
    avg_pairs1.append(pairs1 * (1.0/10))

    y2.append(acc2 * (1.0/10))
    avg_leaves2.append(leaves2 * (1.0/10))
    avg_depth2.append(depth2 * (1.0/10))
    avg_ratio2.append(ratio2 * (1.0/10))
    avg_pairs2.append(pairs2 * (1.0/10))

    y3.append(acc3 * (1.0/10))
    avg_leaves3.append(leaves3 * (1.0/10))
    avg_depth3.append(depth3 * (1.0/10))
    avg_ratio3.append(ratio3 * (1.0/10))
    avg_pairs3.append(pairs3 * (1.0/10))

print("Running time : ", format(time.time() - start))
f_acc1 = open("acc1", "wb")
f_leaves1 = open("leaves1", "wb")
f_depth1 = open("depth1", "wb")
f_ratio1 = open("ratio1", "wb")
f_pairs1 = open("pairs1", "wb")


f_acc2 = open("acc2", "wb")
f_leaves2 = open("leaves2", "wb")
f_depth2 = open("depth2", "wb")
f_ratio2 = open("ratio2", "wb")
f_pairs2 = open("pairs2", "wb")

f_acc3 = open("acc3", "wb")
f_leaves3 = open("leaves3", "wb")
f_depth3 = open("depth3", "wb")
f_ratio3 = open("ratio3", "wb")
f_pairs3 = open("pairs3", "wb")

pickle.dump(y1, f_acc1)
pickle.dump(avg_leaves1, f_leaves1)
pickle.dump(avg_depth1, f_depth1)
pickle.dump(avg_ratio1, f_ratio1)
pickle.dump(avg_pairs1, f_pairs1)

pickle.dump(y2, f_acc2)
pickle.dump(avg_leaves2, f_leaves2)
pickle.dump(avg_depth2, f_depth2)
pickle.dump(avg_ratio2, f_ratio2)
pickle.dump(avg_pairs2, f_pairs2)

pickle.dump(y3, f_acc3)
pickle.dump(avg_leaves3, f_leaves3)
pickle.dump(avg_depth3, f_depth3)
pickle.dump(avg_ratio3, f_ratio3)
pickle.dump(avg_pairs3, f_pairs3)

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
