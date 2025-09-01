import fiftyone as fo

ds = fo.load_dataset("coco-2017-train-foz") # download_coco.py에서 지정한 데이터셋 이름
view = ds.load_saved_view("coco_test01")    # download_coco.py에서 지정한 뷰 이름

view.export(
    export_dir="/mnt/d/workspace/VL/data/coco/fiftyone/coco_test01",    # 내보낼 경로
    dataset_type=fo.types.COCODetectionDataset,                         # COCO 형식으로 내보내기
    label_field="ground_truth",                                         # 레이블 필드 이름
    labels_path="annotations.json",                                     # 레이블 JSON 파일 이름
    export_media=True,                                                  # 이미지도 내보내기
    overwrite=True
)