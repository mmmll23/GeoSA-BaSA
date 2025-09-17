flair_type = "FLAIRDataset"
flair_root = "./data/flair/train/"
flair_test_root = "./data/flair/test/"

flair_crop_size = (512, 512)
flair_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", scale=(512, 512)),
    dict(type="RandomCrop", crop_size=flair_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
flair_test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(512, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
train_flair = dict(
    type=flair_type,
    data_root=flair_root,
    data_prefix=dict(
        img_path="image",
        seg_map_path="label",
    ),
    img_suffix=".png",
    seg_map_suffix=".png",
    pipeline=flair_train_pipeline,
)


test_flair = dict(
    type=flair_type,
    data_root=flair_test_root,
    data_prefix=dict(
        img_path="image",
        seg_map_path="label",
    ),
    img_suffix=".png",
    seg_map_suffix=".png",
    pipeline=flair_test_pipeline,
)