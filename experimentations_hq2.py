from utils import *
import pickle
import sys

g = One_minus()
h = Square_root()

f = Avgdsr()
hq = Gdm(h, g, f)

start = time.time()

dataset, t = generate_2Ddataset(0, 2, 1000, 0, 0.1, [[-10, 10], [-10, 10]])

print("noise (hq): ", 0.05 * int(sys.argv[2]))

# adding noise to the dataset
for k in range(int(sys.argv[2])):
    dataset = add_noise(dataset, 0.05)

sets = get_ten_folds(dataset)

nb_examples = 0

for i in range(10):
    test_set = sets[i]
    train_set = LabeledSet(2)
    for j in range(10):
        if i != j:
            train_set.addExamples(sets[j].x, sets[j].y)
    tree = RDMT(hq, "shannon", 0, 100, 0.01, [1, 2])
    tree.train(train_set)

    print("(hq) BEGIN get_total_examples_ratio : ", time.time())
    nb_examples += tree.get_total_examples_ratio()
    print("(hq) END get_total_examples_ratio : ", time.time())

    print("Iter {} tree (hq)".format(i))

nb_examples = nb_examples * (1.0/10)

print("Running time (hq) (" + sys.argv[2]+ ") : " + str(time.time() - start))
f_nb_examples = open("nb_examples7_"+ sys.argv[2], "wb")

pickle.dump(nb_examples, f_nb_examples)

f_nb_examples.close()

