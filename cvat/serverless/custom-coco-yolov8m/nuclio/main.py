# main.py (핵심 부분만)
import base64, io, json, os
from PIL import Image
from model_handler import ModelHandler

_handler = None

def init_context(context):
    global _handler
    mp = os.getenv("MODEL_PATH", "/opt/nuclio/yolov8m.onnx")
    sz = int(os.getenv("INPUT_SIZE", "640"))
    conf = float(os.getenv("CONF_THRESHOLD", "0.25"))
    iou  = float(os.getenv("IOU_THRESHOLD",  "0.5"))
    _handler = ModelHandler(mp, input_size=sz, conf_thres=conf, iou_thres=iou, logger=context.logger)
    context.logger.info(f"Loaded ONNX model: {mp}")

def _get_payload(event):
    b = event.body
    if isinstance(b, (bytes, bytearray)):
        return json.loads(b.decode("utf-8", errors="ignore"))
    if isinstance(b, str):
        return json.loads(b)
    return b  # already dict

def handler(context, event):
    global _handler
    if _handler is None:
        init_context(context)

    data = _get_payload(event)
    img_b64 = data.get("image")
    if not img_b64:
        return context.Response(body="missing 'image'", status_code=400)

    img = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
    thr = float(data.get("threshold", _handler.conf_thres))

    preds = _handler.infer(img, conf_threshold=thr)
    return context.Response(body=json.dumps(preds), content_type="application/json", status_code=200)
