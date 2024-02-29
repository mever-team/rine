from src.utils import get_transforms, train_one_experiment
from results import best_configs

device = "cuda:0"
workers = 12
ncls_list = [1, 2, 4]
epochs_reduce_lr = [6, 11]
transforms_train, transforms_test, _ = get_transforms()

for ncls in ncls_list:
    experiment = best_configs(ncls=ncls, nbest=1, nepochs=1, showtxt=False)[0]

    # w/o contrastive
    print(f"{ncls}-class: contrastive")
    train_one_experiment(
        experiment=experiment["config"],
        epochss=[experiment["epochs"]],
        epochs_reduce_lr=epochs_reduce_lr,
        transforms_train=transforms_train,
        transforms_val=transforms_test,
        transforms_test=transforms_test,
        workers=workers,
        device=device,
        without="contrastive",
        store=False,
    )

    # w/o alpha
    print(f"{ncls}-class: alpha")
    train_one_experiment(
        experiment=experiment["config"],
        epochss=[experiment["epochs"]],
        epochs_reduce_lr=epochs_reduce_lr,
        transforms_train=transforms_train,
        transforms_val=transforms_test,
        transforms_test=transforms_test,
        workers=workers,
        device=device,
        without="alpha",
        store=False,
    )

    # w/o intermediate features (use only the last transformer block)
    print(f"{ncls}-class: intermediate")
    train_one_experiment(
        experiment=experiment["config"],
        epochss=[experiment["epochs"]],
        epochs_reduce_lr=epochs_reduce_lr,
        transforms_train=transforms_train,
        transforms_val=transforms_test,
        transforms_test=transforms_test,
        workers=workers,
        device=device,
        without="intermediate",
        store=False,
    )
