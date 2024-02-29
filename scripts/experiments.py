from src.utils import get_transforms, train_one_experiment

device = "cuda:0"
workers = 12
epochss = [1]
epochs_reduce_lr = [6, 11]
backbones = [("ViT-L/14", 1024), ("ViT-B/32", 768)]
classess = [
    ["horse"],
    ["chair", "horse"],
    ["car", "cat", "chair", "horse"],
]
factors = [0.1, 0.2, 0.4, 0.8]
nprojs = [1, 2, 4]
proj_dims = [128, 256, 512, 1024]
batch_sizes = [128]
lrs = [1e-3]
experiments = []
for backbone in backbones:
    for classes in classess:
        for factor in factors:
            for nproj in nprojs:
                for proj_dim in proj_dims:
                    for batch_size in batch_sizes:
                        for lr in lrs:
                            experiments.append(
                                {
                                    "training_set": "progan",
                                    "backbone": backbone,
                                    "classes": classes,
                                    "factor": factor,
                                    "nproj": nproj,
                                    "proj_dim": proj_dim,
                                    "batch_size": batch_size,
                                    "lr": lr,
                                    "savpath": f"results/grid/{backbone[0].replace('/', '-')}_{len(classes)}_{factor}_{nproj}_{proj_dim}_{batch_size}_{lr}",
                                }
                            )
transforms_train, transforms_test, _ = get_transforms()

for experiment in experiments:
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
    )
