import torch


class Config:
    n_workers = 8
    batch_size = 8
    epochs = 40
    lr = 2e-4
    verbose = True
    verbose_step = 1
    step_scheduler = False
    validation_scheduler = True
    Scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(mode='min',
                            factor=0.5,
                            patience=0,
                            verbose=False,
                            threshold=0.0001,
                            threshold_mode='abs',
                            cooldown=0,
                            min_lr=5e-9,
                            eps=1e-08)


