import torch
import sys
from pathlib import Path
import os 
from yolov5.utils.dataloaders import LoadImages
from yolov5.utils.general import (Profile, check_img_size,  cv2,
                           non_max_suppression, scale_boxes, )
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolov5.models.common import DetectMultiBackend
import youtube_dl

from timelinebuilder import TimelineBuilder
import json

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model

    device = 0 if torch.cuda.is_available() else -1
    model = DetectMultiBackend(ROOT / 'best.pt', device=device, dnn=False, data=ROOT / 'data/coco128.yaml', fp16=False)

class FilenameCollectorPP(youtube_dl.postprocessor.common.PostProcessor):
    def __init__(self):
        super(FilenameCollectorPP, self).__init__(None)
        self.filenames = []

    def run(self, information):
        self.filenames.append(information["filepath"])
        return [], information


def download_video(url):
    try:
        fname_collector = FilenameCollectorPP()
        ydl.add_post_processor(fname_collector)
        with youtube_dl.YoutubeDL({'f': '134'}) as ydl:
            ydl.download([url])
        return fname_collector.filenames[0]
    except Exception:
        raise Exception("error downloading video")

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    
    # Parse out your arguments
    url = model_inputs.get('url', None)
    if url == None:
        return {'message': "No url provided"}
    vid_stride = model_inputs.get('vid_stride', 3)
    #params
    ss=0
    imgsz=(640, 640)  # inference size (height, width)
    conf_thres=0.55  # confidence threshold
    iou_thres=0.45  # NMS IOU threshold
    max_det=10
    source = download_video(url)

    # Run the model
    # result = model(prompt)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(640, s=stride)  # check image size


    # Dataloader
    bs = 1  # batch_size
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    timelineBuilder = TimelineBuilder()

    for path, im, im0s, vid_cap, s in dataset:
        duration = vid_cap.get(cv2.CAP_PROP_POS_MSEC)
        if duration/1000 < ss:
            continue
        frames = vid_cap.get(cv2.CAP_PROP_POS_FRAMES)
        # print(f"frame_number {vid_cap.get(cv2.CAP_PROP_POS_FRAMES)} {vid_cap.get(cv2.CAP_PROP_POS_MSEC)//1000}")
        s = " ".join(s.split(" ")[:-2])
        s += f" {duration//1000} {frames}: "
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            # print(im.shape)

            # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=False, visualize=False)
        
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, None, None, max_det=max_det)
        
        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            # save_path = str(save_dir / p.name)  # im.jpg
            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # imc = im0.copy() if save_crop else im0  # for save_crop
            # annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
            # print(s)
            timelineBuilder.append(s, frames, duration)

    # Return the results as a dictionary
    return {"events": json.dumps(timelineBuilder.timeline)}
