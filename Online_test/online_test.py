from torch.utils.data import DataLoader

from continuum import ClassIncremental
from continuum.datasets import CIFAR10
from continuum.tasks import split_train_val

dataset = CIFAR10("/temp", download=True, train=True)
scenario = ClassIncremental(
    dataset,
    increment=5,
    initial_increment=5
)

print(f"Number of classes: {scenario.nb_classes}.")
print(f"Number of tasks: {scenario.nb_tasks}.")
print(f"scenario",len(scenario))  #* 2
# print(f"scenario",scenario)
for task_id, train_taskset in enumerate(scenario):
    train_taskset, val_taskset = split_train_val(train_taskset, val_split=0.1)
    train_loader = DataLoader(train_taskset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_taskset, batch_size=32, shuffle=True)

    for x, y, t in train_loader:
        # Do your cool stuff here
        pass