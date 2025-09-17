import os, io, json, base64, re
from typing import List, Dict, Any
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "/opt/nuclio/model/yolo11m-pose.pt")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))
KP_MODE = os.getenv("KP_MODE", "12")  # "12" or "17"
LABEL_NAME = os.getenv("LABEL_NAME", "person")  # CVAT 스펙과 반드시 동일

SELECT_IDX_12 = [5,6,7,8,9,10,11,12,13,14,15,16]
KEYPOINT_NAMES_12 = [str(i) for i in range(1, 13)]
KEYPOINT_NAMES_17 = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle"
]

_model = None

def init_context(context):
    global _model
    context.logger.info(f"Loading YOLO pose model from {MODEL_PATH} ...")
    _model = YOLO(MODEL_PATH)
    context.logger.info("Model loaded. Ready.")

def _read_image_from_event(event) -> Image.Image:
    headers = {str(k).lower(): str(v) for k, v in getattr(event, "headers", {}).items()} if getattr(event, "headers", None) else {}
    ctype = headers.get("content-type", "").lower()
    body = event.body or b""

    # 1) raw binary / image/*
    if "application/octet-stream" in ctype or ctype.startswith("image/"):
        return Image.open(io.BytesIO(body)).convert("RGB")

    # 2) JSON base64 {"image": "..."} / {"data": "..."} (data URI 허용)
    if "application/json" in ctype or not ctype:
        try:
            obj = body if isinstance(body, dict) else json.loads(body if isinstance(body, str) else body.decode("utf-8", "ignore"))
            if isinstance(obj, dict):
                b64 = obj.get("image") or obj.get("data")
                if isinstance(b64, str):
                    if "," in b64 and ";base64" in b64.split(",", 1)[0]:
                        b64 = b64.split(",", 1)[1]
                    return Image.open(io.BytesIO(base64.b64decode(b64, validate=False))).convert("RGB")
        except Exception:
            pass

    # 3) multipart/form-data: field name "image"
    if "multipart/form-data" in ctype:
        m = re.search(r"boundary=([^;]+)", ctype)
        if not m:
            raise ValueError("multipart boundary not found")
        boundary = m.group(1).encode()
        for part in body.split(b"--" + boundary):
            if b"Content-Disposition" in part and b'name="image"' in part and b"filename=" in part:
                if b"\r\n\r\n" in part:
                    filedata = part.split(b"\r\n\r\n", 1)[1].rsplit(b"\r\n", 1)[0]
                    return Image.open(io.BytesIO(filedata)).convert("RGB")

    raise ValueError("Cannot parse input image from request")

def _kp_names():
    return KEYPOINT_NAMES_12 if KP_MODE.strip() == "12" else KEYPOINT_NAMES_17

def _make_skeleton_object(kps_xy, kps_conf, label_name: str) -> Dict[str, Any]:
    elements = []
    names = _kp_names()
    for i, name in enumerate(names):
        x = float(kps_xy[i][0]); y = float(kps_xy[i][1])
        conf = float(kps_conf[i]) if kps_conf is not None else 1.0
        elements.append({
            "type": "points",
            "label": name,            # 12k는 "1".."12"
            "points": [x, y],
            "outside": bool(conf < 0.5),
            "occluded": False,
            "attributes": []
        })
    return {"type": "skeleton", "label": label_name, "elements": elements, "attributes": []}

def handler(context, event):
    if _model is None:
        init_context(context)

    # 400으로 깔끔히 반환
    try:
        image = _read_image_from_event(event)
    except ValueError as e:
        return context.Response(body=str(e).encode("utf-8"),
                                headers={"Content-Type": "text/plain"},
                                status_code=400)

    results = _model.predict(source=image, conf=CONF_THRESHOLD, verbose=False)

    objects: List[Dict[str, Any]] = []
    for r in results:
        if r.keypoints is None:
            continue
        kxy = r.keypoints.xy
        kcf = r.keypoints.conf
        n_det = len(kxy)

        # boxes confidence (있으면)
        det_confs_list = None
        if getattr(r, "boxes", None) is not None and getattr(r.boxes, "conf", None) is not None:
            try:
                det_confs_list = r.boxes.conf.reshape(-1).detach().cpu().tolist()
            except Exception:
                pass

        for det_i in range(n_det):
            pts = kxy[det_i]
            cfs = kcf[det_i] if kcf is not None else None
            if KP_MODE.strip() == "12":
                pts = pts[SELECT_IDX_12]
                cfs = cfs[SELECT_IDX_12] if cfs is not None else None

            obj = _make_skeleton_object(pts, cfs, label_name=LABEL_NAME)
            if det_confs_list and det_i < len(det_confs_list):
                try:
                    obj["confidence"] = float(det_confs_list[det_i])
                except Exception:
                    pass
            objects.append(obj)

    # 최상위에 리스트를 바로 반환 (빈 리스트여도 OK)
    return context.Response(
        body=json.dumps(objects).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        status_code=200,
    )
