import numpy as np


def mutation(prob, best_score, population, population_scores):
    insertion_size = 5
    learn_rate = [0.01, 0.01, 0.01, 0.02]
    species = []
    best = list(population[np.argsort(population_scores)[-1]])

    # pick the largest 5 individuals to perform
    for _index, _i in enumerate(population):
        mut_temp = np.random.rand()
        if mut_temp < prob:  # perform mutation, else pass
            _score = population_scores[_index]
            weight = 4 * (np.abs(_score - best_score) / best_score)  # 0 <= weight < 5
            _new_individual = []
            # alphas should < 0
            for _j, _par_a in enumerate(_i[:2]):
                _gain = 2 * (np.random.rand() - 0.5) * ((1 + weight) * learn_rate[_j])  # step (1, 5) of learn rate
                # proportional to the parameter size
                while _par_a + _gain > 0:
                    _gain = 2 * (np.random.rand() - 0.5) * ((1 + weight) * learn_rate[_j])
                _par_a += _gain  # update parameter
                _new_individual.append(_par_a)
            # betas should >= 0
            for _k, _par_b in enumerate(_i[2:]):
                _gain = 2 * (np.random.rand() - 0.5) * ((1 + weight) * learn_rate[2 + _k])  # step (1, 5) of learn rate
                # proportional to the parameter size
                while _par_b + _gain < 0:
                    _gain = 2 * (np.random.rand() - 0.5) * ((1 + weight) * learn_rate[2 + _k])
                _par_b += _gain  # update parameter
                _new_individual.append(_par_b)
            species.append(_new_individual)
        else:
            species.append(_i)
    # insert the best solution so far
    """ always preserve the best solution """
    species.extend(insertion_size * [best])
    return species


prob_mut = 0.8
# %% define s
s = [[-0.33396139344550824,
      -0.05709201446414879,
      0.183723748729145,
      0.1343333710625816],
     [-0.33396139344550824,
      -0.05709201446414879,
      0.183723748729145,
      0.1343333710625816],
     [-0.33396139344550824,
      -0.05709201446414879,
      0.183723748729145,
      0.1343333710625816],
     [-0.33396139344550824,
      -0.05709201446414879,
      0.183723748729145,
      0.1343333710625816],
     [-0.33396139344550824,
      -0.05709201446414879,
      0.183723748729145,
      0.1343333710625816],
     [-0.33396139344550824,
      -0.05709201446414879,
      0.183723748729145,
      0.1343333710625816],
     [-0.33396139344550824,
      -0.05709201446414879,
      0.183723748729145,
      0.1343333710625816],
     [-0.33396139344550824,
      -0.05709201446414879,
      0.183723748729145,
      0.1343333710625816],
     [-0.33396139344550824,
      -0.05709201446414879,
      0.183723748729145,
      0.1343333710625816],
     [-0.33396139344550824,
      -0.05709201446414879,
      0.183723748729145,
      0.1343333710625816]]
# %%
species_1 = [[-0.37974139765941783, -0.2558939179688775, 0.04893578316589828, 0.05210324618171788],
             [-0.08918215758059533, -0.24405051342668937, 0.027020415460782002, 0.2056606245278888],
             [-0.16167280374767834, -0.2411843920976503, 0.03404410015008346, 0.3076044553748146],
             [-0.46797554664851887, -0.08691471688216373, 0.27465618122012814, 0.8535210297561443],
             [-0.16654700848822268, -0.0887516253882134, 0.14708878950043483, 0.3303207960587167],
             [-0.3236607278310718, -0.0668914251165349, 0.19367692132502703, 0.4580954274520535],
             [-0.3232875461086227, -0.028247387947372693, 0.24030778479735282, 0.3322569213448343],
             [-0.059277941835195136, -0.4592104661803278, 0.241806890829197, 0.43319110214340956],
             [-0.05, -0.05, 0.03, 0.1], [-0.03, -0.01, 0.02, 0.1]]

Indices = [2, 4, 3, 3, 1, 1, 5, 6, 6, 6]

species_selected = [species_1[_] for _ in Indices]

species_selected_kakunin = [[-0.16167280374767834, -0.2411843920976503, 0.03404410015008346, 0.3076044553748146],
                            [-0.16654700848822268, -0.0887516253882134, 0.14708878950043483, 0.3303207960587167],
                            [-0.46797554664851887, -0.08691471688216373, 0.27465618122012814, 0.8535210297561443],
                            [-0.46797554664851887, -0.08691471688216373, 0.27465618122012814, 0.8535210297561443],
                            [-0.08918215758059533, -0.24405051342668937, 0.027020415460782002, 0.2056606245278888],
                            [-0.08918215758059533, -0.24405051342668937, 0.027020415460782002, 0.2056606245278888],
                            [-0.3236607278310718, -0.0668914251165349, 0.19367692132502703, 0.4580954274520535],
                            [-0.3232875461086227, -0.028247387947372693, 0.24030778479735282, 0.3322569213448343],
                            [-0.3232875461086227, -0.028247387947372693, 0.24030778479735282, 0.3322569213448343],
                            [-0.3232875461086227, -0.028247387947372693, 0.24030778479735282, 0.3322569213448343]]

scores = [1.791076859544893e-05, 1.2681865598677504e-05, 1.4428350964154472e-05, 1.4428350964154472e-05,
          1.5858954373553838e-05, 1.5858954373553838e-05, 1.8423138530265e-05, 1.9820928956103915e-05,
          1.9820928956103915e-05, 1.9820928956103915e-05]

para_record = max(scores)

species_after_mutate = [[-0.15935079281291667, -0.2392587596852932, 0.044838800992814454, 0.30931614995758644],
                        [-0.16368112780560284, -0.07208612687039126, 0.1374833482478515, 0.33512637745697726],
                        [-0.4868034912898777, -0.07104605119360258, 0.25303827252555083, 0.8620613311717086],
                        [-0.4868034912898777, -0.07104605119360258, 0.25303827252555083, 0.8620613311717086],
                        [-0.0762932173461917, -0.24203346114202395, 0.030322018758716866, 0.19252402242462965],
                        [-0.0762932173461917, -0.24203346114202395, 0.030322018758716866, 0.19252402242462965],
                        [-0.3286247099941957, -0.05514039485630604, 0.19082206390597659, 0.4533346889566321],
                        [-0.3326928160827272, -0.019886069734282426, 0.24059796591564855, 0.29567647677240744],
                        [-0.3326928160827272, -0.019886069734282426, 0.24059796591564855, 0.29567647677240744],
                        [-0.3326928160827272, -0.019886069734282426, 0.24059796591564855, 0.29567647677240744],
                        [-0.3232875461086227, -0.028247387947372693, 0.24030778479735282, 0.3322569213448343],
                        [-0.3232875461086227, -0.028247387947372693, 0.24030778479735282, 0.3322569213448343],
                        [-0.3232875461086227, -0.028247387947372693, 0.24030778479735282, 0.3322569213448343],
                        [-0.3232875461086227, -0.028247387947372693, 0.24030778479735282, 0.3322569213448343],
                        [-0.3232875461086227, -0.028247387947372693, 0.24030778479735282, 0.3322569213448343]]

foo = mutation(1, para_record, species_selected_kakunin, scores)
# print(foo)