# model_handler.py (교체/업데이트)
import numpy as np, onnxruntime as ort, cv2

COCO = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
"fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
"elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
"skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
"wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog",
"pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote",
"keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
"hair drier","toothbrush"]

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
    def __init__(self, path, input_size=640, conf_thres=0.25, iou_thres=0.5, logger=None):
        self.input_size = int(input_size); self.conf_thres=float(conf_thres); self.iou_thres=float(iou_thres)
        self.logger = logger
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.inp = self.session.get_inputs()[0].name

        # 일부 모델은 이름이 없거나 여러 출력이 있어도 첫 번째만 사용
        self.out = self.session.get_outputs()[0].name

    def _normalize_output(self, out):
        """
        out shape variants we accept:
        (N,84) / (1,N,84) / (84,N) / (1,84,N) / (N,85) / (1,N,85) / (85,N) / (1,85,N)
        Returns: (N, D) with D in {84,85}
        """
        arr = out
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
            # else (N,84/85) 그대로
        else:
            raise RuntimeError(f"Unexpected output shape {arr.shape}")
        return arr

    def infer(self, pil_img, conf_threshold=None):
        if conf_threshold is None: conf_threshold = self.conf_thres
        im0 = np.array(pil_img)[:, :, ::-1]  # RGB->BGR
        img, r, (dx, dy) = _letterbox(im0, self.input_size)
        img = img[:, :, ::-1].astype(np.float32)/255.0  # RGB
        img = np.transpose(img, (2,0,1))[None, ...]

        pred = self.session.run([self.out], {self.inp: img})[0]
        pred = self._normalize_output(pred)  # (N,84 or 85)

        D = pred.shape[1]
        if D == 84:
            boxes_xywh = pred[:, :4]
            cls_scores = pred[:, 4:]
            class_ids = cls_scores.argmax(1)
            scores = cls_scores[np.arange(cls_scores.shape[0]), class_ids]
        elif D == 85:
            # 4 box + 1 obj + 80 cls
            boxes_xywh = pred[:, :4]
            obj = pred[:, 4:5]
            cls_scores = pred[:, 5:]
            class_ids = cls_scores.argmax(1)
            scores = (cls_scores[np.arange(cls_scores.shape[0]), class_ids] * obj[:,0])
        else:
            raise RuntimeError(f"Unsupported last dimension {D}")

        m = scores >= conf_threshold
        boxes_xywh, scores, class_ids = boxes_xywh[m], scores[m], class_ids[m]

        # xywh(center) -> xyxy (letterboxed)
        xy = boxes_xywh[:, :2]; wh = boxes_xywh[:, 2:4]
        xyxy = np.concatenate([xy - wh/2, xy + wh/2], 1)

        # 정규화 출력 대응
        if xyxy.max() <= 1.5: xyxy *= self.input_size

        # letterbox 보정 & clip
        xyxy[:, [0,2]] -= dx; xyxy[:, [1,3]] -= dy
        xyxy /= r
        h, w = im0.shape[:2]
        xyxy[:, [0,2]] = np.clip(xyxy[:, [0,2]], 0, w-1)
        xyxy[:, [1,3]] = np.clip(xyxy[:, [1,3]], 0, h-1)

        if xyxy.shape[0]:
            k = _nms(xyxy, scores, self.iou_thres)
            xyxy, scores, class_ids = xyxy[k], scores[k], class_ids[k]

        out=[]
        for box, sc, cid in zip(xyxy, scores, class_ids):
            x1,y1,x2,y2 = box.tolist()
            name = COCO[int(cid)] if 0 <= int(cid) < len(COCO) else str(int(cid))
            out.append({"type":"rectangle","label":name,
                        "points":[float(x1),float(y1),float(x2),float(y2)],
                        "attributes":[], "confidence":float(sc)})
        return out
