whumix_type = "WHUMIXDataset"
whumix_root = "./data/whumix/train/"
whumix_crop_size = (512, 512)
whumix_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", scale=(512, 512)),
    dict(type="RandomCrop", crop_size=whumix_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]

train_whumix = dict(
    type=whumix_type,
    data_root=whumix_root,
    data_prefix=dict(
        img_path="image",
        seg_map_path="label01",
    ),
    img_suffix=".tif",
    seg_map_suffix=".png",
    pipeline=whumix_train_pipeline,
)
