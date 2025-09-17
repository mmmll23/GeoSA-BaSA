oem01_type = "WHUMIXDataset"
oem01_root = "./data/oem/trainval/"
# oem01_crop_size = (512, 512)

oem01_test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(512, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]

val_oem01 = dict(
    type=oem01_type,
    data_root=oem01_root,
    data_prefix=dict(
        img_path="image",
        seg_map_path="label01",
    ),
    img_suffix=".png",
    seg_map_suffix=".png",
    pipeline=oem01_test_pipeline,
)