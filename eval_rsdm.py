from utils import *
import pickle
import sys

g = Log()
h = Sum()

f = Dsr()
rsdm = Gdm(h, g, f)

start = time.time()

dataset, t = generate_2Ddataset(0, 2, 1000, 0, 0.1, [[-10, 10], [-10, 10]])

print("noise (rsdm): ", 0.05 * int(sys.argv[2]))

# adding noise to the dataset
for k in range(int(sys.argv[2])):
    dataset = add_noise(dataset, 0.05)

sets = get_ten_folds(dataset)

evaluation = 0
p_ratio = 0

for i in range(10):
    test_set = sets[i]
    train_set = LabeledSet(2)
    for j in range(10):
        if i != j:
            train_set.addExamples(sets[j].x, sets[j].y)
    tree = RDMT(rsdm, "shannon", 0, 100, 0.01, [1, 2])
    tree.train(train_set)

    print("(rsdm) BEGIN evaluate_monotonicity : ", time.time())
    evaluation += tree.evaluate_monotonicity()
    print("(rsdm) END evaluate_monotonicity : ", time.time())

    print("(rsdm) BEGIN pairs_ratio : ", time.time())
    p_ratio += tree.pairs_ratio()
    print("(rsdm) END pairs_ratio : ", time.time())

    print("Iter {} tree (rsdm)".format(i))


evaluation = evaluation * (1.0/10)
p_ratio = p_ratio * (1.0/10)

print("Running time (rsdm) (" + sys.argv[2]+ ") : " + str(time.time() - start))
f_eval = open("eval1_"+ sys.argv[2], "wb")
f_ratio = open("p_ratio1_"+sys.argv[2], "wb")

pickle.dump(evaluation, f_eval)
pickle.dump(p_ratio, f_ratio)

f_eval.close()
f_ratio.close()
