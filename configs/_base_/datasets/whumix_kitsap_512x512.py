whumix_kitsap_type = "WHUMIXDataset"
whumix_kitsap_val_root = "./data/whumix/kitsap/"

whumix_kitsap_crop_size = (512, 512)

whumix_kitsap_test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(512, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]

val_whumix_kitsap = dict(
    type=whumix_kitsap_type,
    data_root=whumix_kitsap_val_root,
    data_prefix=dict(
        img_path="image",
        seg_map_path="label01",
    ),
    img_suffix=".tif",
    seg_map_suffix=".png",
    pipeline=whumix_kitsap_test_pipeline,
)