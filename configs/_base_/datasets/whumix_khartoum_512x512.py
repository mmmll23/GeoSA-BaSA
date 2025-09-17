whumix_khartoum_type = "WHUMIXDataset"
whumix_khartoum_val_root = "./data/whumix/khartoum/"

whumix_khartoum_crop_size = (512, 512)

whumix_khartoum_test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(512, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]

val_whumix_khartoum = dict(
    type=whumix_khartoum_type,
    data_root=whumix_khartoum_val_root,
    data_prefix=dict(
        img_path="image",
        seg_map_path="label01",
    ),
    img_suffix=".tif",
    seg_map_suffix=".png",
    pipeline=whumix_khartoum_test_pipeline,
)