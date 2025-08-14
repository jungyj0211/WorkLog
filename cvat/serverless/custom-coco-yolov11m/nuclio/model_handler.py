import numpy as np, onnxruntime as ort, cv2

def _letterbox(im, new=640):
    h, w = im.shape[:2]
    if isinstance(new, int): new = (new, new)
    r = min(new[0]/h, new[1]/w)
    nh, nw = int(round(h*r)), int(round(w*r))
    im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = (new[0]-nh)//2, new[0]-nh-(new[0]-nh)//2
    left, right = (new[1]-nw)//2, new[1]-nw-(new[1]-nw)//2
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))
    return im, r, (left, top)

def _iou(box, boxes):
    xx1 = np.maximum(box[0], boxes[:,0]); yy1 = np.maximum(box[1], boxes[:,1])
    xx2 = np.minimum(box[2], boxes[:,2]); yy2 = np.minimum(box[3], boxes[:,3])
    inter = np.maximum(0, xx2-xx1)*np.maximum(0, yy2-yy1)
    a1 = (box[2]-box[0])*(box[3]-box[1]); a2 = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    return inter/(a1+a2-inter+1e-6)

def _nms(xyxy, scores, thr):
    idx = scores.argsort()[::-1]; keep=[]
    while idx.size:
        i = idx[0]; keep.append(i)
        if idx.size==1: break
        idx = idx[1:][_iou(xyxy[i], xyxy[idx[1:]]) < thr]
    return np.array(keep, dtype=np.int32)

class ModelHandler:
    """
    YOLOv8/11 ONNX postprocess.
    - Supports single fused output: (1, 84/85, N) or (1, N, 84/85)
    - Supports multi-output head (P3/P4/P5)
    - Outputs CVAT rectangles with label IDs (1..80)
    """
    def __init__(self, path, input_size=640, conf_thres=0.35, iou_thres=0.5, logger=None):
        self.input_size = int(input_size); self.conf_thres=float(conf_thres); self.iou_thres=float(iou_thres)
        self.logger = logger
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.inp = self.session.get_inputs()[0].name
        self.outs = [o.name for o in self.session.get_outputs()]

    def _normalize_single(self, arr):
        # Returns (N, D) with D in {84, 85}
        if arr.ndim == 3:
            b, a, c = arr.shape
            if b == 1 and c in (84,85):      # (1, N, 84/85)
                arr = arr[0]
            elif b == 1 and a in (84,85):    # (1, 84/85, N)
                arr = arr[0].T
            else:
                raise RuntimeError(f"Unexpected 3D shape {arr.shape}")
        elif arr.ndim == 2:
            if arr.shape[0] in (84,85) and arr.shape[1] > arr.shape[0]:
                arr = arr.T                   # (84/85, N) -> (N, 84/85)
        else:
            raise RuntimeError(f"Unexpected output shape {arr.shape}")
        return arr

    def _normalize_outputs(self, outs):
        if isinstance(outs, list):
            parts = [self._normalize_single(o) for o in outs]
            Dset = {p.shape[1] for p in parts}
            if len(Dset) != 1:
                raise RuntimeError(f"Inconsistent output dims {Dset}")
            return np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
        else:
            return self._normalize_single(outs)

    def infer(self, pil_img, conf_threshold=None):
        if conf_threshold is None: conf_threshold = self.conf_thres
        im0 = np.array(pil_img)[:, :, ::-1]  # RGB->BGR
        img, r, (dx, dy) = _letterbox(im0, self.input_size)
        img = img[:, :, ::-1].astype(np.float32)/255.0  # RGB
        img = np.transpose(img, (2,0,1))[None, ...]

        outs = self.session.run(self.outs, {self.inp: img})
        if len(outs) == 1:
            outs = outs[0]
        pred = self._normalize_outputs(outs)  # (N,84 or 85)

        D = pred.shape[1]
        if D == 84:
            boxes_xywh = pred[:, :4]
            cls_scores = pred[:, 4:]
            class_ids = cls_scores.argmax(1)
            scores = cls_scores[np.arange(cls_scores.shape[0]), class_ids]
        elif D == 85:
            boxes_xywh = pred[:, :4]
            obj = pred[:, 4:5]
            cls_scores = pred[:, 5:]
            class_ids = cls_scores.argmax(1)
            scores = cls_scores[np.arange(cls_scores.shape[0]), class_ids] * obj[:,0]
        else:
            raise RuntimeError(f"Unsupported last dimension {D}")

        m = scores >= conf_threshold
        boxes_xywh, scores, class_ids = boxes_xywh[m], scores[m], class_ids[m]

        xy = boxes_xywh[:, :2]; wh = boxes_xywh[:, 2:4]
        xyxy = np.concatenate([xy - wh/2, xy + wh/2], 1)

        if xyxy.size and xyxy.max() <= 1.5:
            xyxy *= self.input_size  # normalized

        xyxy[:, [0,2]] -= dx; xyxy[:, [1,3]] -= dy
        xyxy /= r
        h, w = im0.shape[:2]
        xyxy[:, [0,2]] = np.clip(xyxy[:, [0,2]], 0, w-1)
        xyxy[:, [1,3]] = np.clip(xyxy[:, [1,3]], 0, h-1)

        if xyxy.shape[0]:
            k = _nms(xyxy, scores, self.iou_thres)
            xyxy, scores, class_ids = xyxy[k], scores[k], class_ids[k]

        out=[]
        COCO = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
        "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
        "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
        "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
        "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog",
        "pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote",
        "keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
        "hair drier","toothbrush"]

        for box, sc, cid in zip(xyxy, scores, class_ids):
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            cid = int(cid)
            label_id = cid + 1                      # 1..80
            label_name = COCO[cid] if 0 <= cid < len(COCO) else str(label_id)
            out.append({
                "type": "rectangle",
                "label_id": label_id,               # ← 정수 ID
                "label": label_name,                # ← 이름도 같이
                "points": [x1, y1, x2, y2],
                "attributes": [],
                "confidence": float(sc),
            })
        return out
