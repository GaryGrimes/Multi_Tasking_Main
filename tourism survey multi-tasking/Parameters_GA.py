import matplotlib.pyplot as plt
import numpy as np


def penalty(particle):
    _answer = [-0.02, -0.01, 0.3, 0.1]
    diff = np.array(_answer) - np.array(particle)
    _penalty = np.exp(np.linalg.norm(diff))
    return _penalty


def selection(population, s_size):
    scores = []
    for _individual in population:
        scores.append(1 / penalty(_individual) ** 5)
    best_one_idx = np.argsort(scores)[-1]
    f_sum = sum(scores)
    prob = [_ / f_sum for _ in scores]
    # calculate accumulated prob
    prob_acu = [sum(prob[:_]) + prob[_] for _ in range(len(prob))]
    prob_acu[-1] = 1

    # return selected idx
    indices = []
    for _ in range(s_size - 10):
        indices.append(next(x[0] for x in enumerate(prob_acu) if x[1] > np.random.rand()))
    indices.extend(10 * [best_one_idx])
    return indices


def mutation(p, best_score, population, scores):
    learn_rate = [0.01, 0.01, 0.01, 0.02]
    species = []
    best = list(population[np.argsort(scores)[-1]])
    for _idx, _i in enumerate(population):
        if np.random.rand() < p:  # perform mutation, else pass
            _score = scores[_idx]
            weight = 4 * (np.abs(_score - best_score) / best_score)  # 0 <= weight < 5
            # alphas should < 0
            for _j, _par in enumerate(_i[:2]):
                _gain = 2 * (np.random.rand() - 0.5) * ((1 + weight) * learn_rate[_j])  # step (1, 5) of learn rate
                # proportional to the parameter size
                while _par + _gain > 0:
                    _gain = 2 * (np.random.rand() - 0.5) * ((1 + weight) * learn_rate[_j])
                _par = _par + _gain  # update parameter
                _i[_j] = _par
            # betas should >= 0
            for _j, _par in enumerate(_i[2:]):
                _gain = 2 * (np.random.rand() - 0.5) * ((1 + weight) * learn_rate[2 + _j])  # step (1, 5) of learn rate
                # proportional to the parameter size
                while _par + _gain < 0:
                    _gain = 2 * (np.random.rand() - 0.5) * ((1 + weight) * learn_rate[2 + _j])
                _par = _par + _gain  # update parameter
                _i[2 + _j] = _par
        species.append(_i)
    # insert the best solution so far
    """ always preserve the best solution """
    species.extend(10 * [best])
    return species


inn = 50
itr_max = 500
prob_mut = 0.8  # parameter update probability

s = []  # species
# generate first population
for i in range(inn):
    # random alphas
    a1, a2 = np.random.uniform(-0.5, -0.01), np.random.uniform(-0.5, -0.01)
    # random betas
    b2, b3 = np.random.uniform(0.01, 0.3), np.random.uniform(0.02, 1)
    s.append([a1, a2, b2, b3])

ymean, ymax, xmax = [], [], []  # 记录平均score, 每一世代max score， 每世代最佳个体
gnrmax = []
"""
# manually insert reasonable parameters
s[-2], s[-1] = [-0.05, -0.05, 0.03, 0.1], [-0.03, -0.01, 0.02, 0.1]
"""

record = max([1 / penalty(_) ** 5 for _ in s])

for itr in range(itr_max):
    # for idx, individual in enumerate(s):
    #     print('{}: penalty: {}'.format(idx, penalty(individual)))

    #  print (record)
    Indices = selection(s, inn)

    # selection
    s = list(s[_] for _ in Indices)

    # calculate scores and set record
    score = [1 / penalty(_) ** 5 for _ in s]
    Best_score = max(score)
    if Best_score > record:
        record = Best_score
    gnrmax.append(Best_score)
    ymean.append(np.mean(score))
    ymax.append(record)
    xmax.append(s[np.argsort(score)[-1]])

    # mutation
    s = mutation(prob_mut, record, s, score)

    # save results


# %% plot

x = range(1, itr_max + 1)
fig = plt.figure(dpi=150)
ax = fig.add_subplot(111)
ax.plot(x, ymean, color='lightblue')
ax.plot(ymax, color='xkcd:orange')
plt.xlabel("# of iterations")
# 显示纵轴标签
plt.ylabel("score")
plt.xlim([0, itr_max])
plt.ylim([0, 1])
# 显示图标题
plt.title("Parameter update process")
plt.legend(['average', 'best'], loc='best')
plt.show()

# TODO score 应为 1/penalty

# TODO 计算每一组参数的objective func的时候，用dict存储当memo. if parameters not in memo, 写入。然后从memo里读取
