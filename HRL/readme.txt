meta agent: has 2 arms datasets
choosing an arm means:
1 take batch from dataset.
2 use offline QL to learn from batch for 100 train episodes.
3 run offline QL agent and take avg reward for 10 episodes.
4 return this reward as arm reward.
5 reset offline QL agent.

for phase 2:
Something's need to be added. some candidates are:
1 Make meta agent decision-making sequential instead of bandit. State specification required.
2 Add features i.e. intrinsic reward.
2 Increase datasets.
3 Make action space continuous.
    instead of arms or discrete actions, your action will be a mix proportion of datasets.
    this makes actions: S -> delta(datasets)
    policy network (actor). should use policy gradient.
    can use Boltzmann distribution: a1:x1,..., an:xn: p(ai) = exp(xi/tau)/ (sum_j exp(xj/tau))
4 Examine transitions instead of datasets

