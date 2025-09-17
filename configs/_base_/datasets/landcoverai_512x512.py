landcoverai_type = "LandcoveraiDataset"
landcoverai_root = "./data/landcoverai/landcoverai025/"
landcoverai_test_root = "./data/landcoverai/landcoverai05/"

landcoverai_crop_size = (512, 512)
landcoverai_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", scale=(512, 512)),
    dict(type="RandomCrop", crop_size=landcoverai_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
landcoverai_test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(512, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
train_landcoverai = dict(
    type=landcoverai_type,
    data_root=landcoverai_root,
    data_prefix=dict(
        img_path="image",
        seg_map_path="label",
    ),
    img_suffix=".png",
    seg_map_suffix=".png",
    pipeline=landcoverai_train_pipeline,
)

test_landcoverai = dict(
    type=landcoverai_type,
    data_root=landcoverai_test_root,
    data_prefix=dict(
        img_path="image",
        seg_map_path="label",
    ),
    img_suffix=".png",
    seg_map_suffix=".png",
    pipeline=landcoverai_test_pipeline,
)