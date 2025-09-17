whumix_dunedin_type = "WHUMIXDataset"
whumix_dunedin_val_root = "./data/whumix/dunedin/"

whumix_dunedin_crop_size = (512, 512)

whumix_dunedin_test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(512, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]

val_whumix_dunedin = dict(
    type=whumix_dunedin_type,
    data_root=whumix_dunedin_val_root,
    data_prefix=dict(
        img_path="image",
        seg_map_path="label01",
    ),
    img_suffix=".tif",
    seg_map_suffix=".png",
    pipeline=whumix_dunedin_test_pipeline,
)