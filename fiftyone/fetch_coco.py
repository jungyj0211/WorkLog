# fetch_coco.py
# - FiftyOne Dataset Zoo를 이용해 COCO-2017을 다운로드/로딩하고 앱을 실행한다.

# 사용법(WSL 터미널):
#     conda activate venv_name      # 또는 사용하는 venv 활성화
#     # 필요한 경우: pip install -U fiftyone
#     python fetch_coco.py

# 옵션 수정은 아래 USER SETTINGS 섹션을 편집

import fiftyone as fo
import fiftyone.zoo as foz

# ===== USER SETTINGS =====
DATASET_DIR = "/mnt/d/workspace/VL/fiftyone/test"  # 다운로드/캐시 저장 경로
SPLIT = "train"                             # "train" 또는 "validation"
LABEL_TYPES = ["detections"]                # ["detections","segmentations","keypoints"] 등
CLASSES = ["person"]                        # 예: ["person","car"]  -> 일부 클래스만 받기
MAX_SAMPLES = 2000                          # 예: 200  -> 샘플 제한, None이면 전체
DATASET_NAME = f"coco-2017-{SPLIT}-foz"     # FiftyOne 내부 데이터셋 이름
# ==========================

print("[INFO] Loading from FiftyOne Dataset Zoo (this may download if missing) ...")
ds = foz.load_zoo_dataset(
    "coco-2017",
    split=SPLIT,
    label_types=LABEL_TYPES,
    classes=CLASSES,
    max_samples=MAX_SAMPLES,
    # dataset_dir=DATASET_DIR,  # 캐시 위치 지정 (기본값 사용 시 주석 처리), 에러 발생 시 주석 처리
    dataset_name=DATASET_NAME,
    overwrite=True,
    shuffle=True,
    seed=42,
)

print(f"[INFO] Dataset loaded: {ds.name} with {len(ds)} samples")
print("[INFO] Launching FiftyOne App on http://127.0.0.1:5151 ...")
session = fo.launch_app(ds)
session.wait()