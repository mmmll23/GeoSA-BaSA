_base_ = [
    "./whumix_512x512.py",
    "./whumix_kitsap_512x512.py",
    "./whumix_wuxi_512x512.py",
    "./whumix_dunedin_512x512.py",
    "./whumix_potsdam_512x512.py",
    "./whumix_khartoum_512x512.py",
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset={{_base_.train_whumix}},
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="ConcatDataset",
        datasets=[
            {{_base_.val_whumix_kitsap}},
            {{_base_.val_whumix_wuxi}},
            {{_base_.val_whumix_dunedin}},
            {{_base_.val_whumix_potsdam}},
            {{_base_.val_whumix_khartoum}},
        ],
    ),
)
test_dataloader = val_dataloader
val_evaluator = dict(
    type="DGIoUMetric", iou_metrics=["mIoU","mFscore"], dataset_keys=["kitsap", "wuxi", "dunedin", "potsdam", "khartoum"]
)
test_evaluator=val_evaluator
