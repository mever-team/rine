from src.utils import get_transforms, train_one_experiment
from results import best_configs

device = "cuda:0"
workers = 12
epochss = [1]
epochs_reduce_lr = [6, 11]
experiments = [
    x["config"]
    for ncls in [1, 2, 4]
    for x in best_configs(ncls=ncls, nbest=1, nepochs=1, showtxt=False)
]

transforms_train, transforms_test, _ = get_transforms()

for experiment in experiments:
    for ds_frac in [0.2, 0.5, 0.8, 1.0]:
        print(f"training data fraction: {ds_frac}")
        train_one_experiment(
            experiment=experiment,
            epochss=epochss,
            epochs_reduce_lr=epochs_reduce_lr,
            transforms_train=transforms_train,
            transforms_val=transforms_test,
            transforms_test=transforms_test,
            workers=workers,
            device=device,
            without=None,
            store=False,
            ds_frac=ds_frac,
        )
