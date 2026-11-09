"""Unlabelled geometric proposal for the small central BODY object, with fallback."""
import numpy as np
from PIL import Image,ImageOps
from scipy import ndimage


def prepare_body(image):
    image=image.convert('RGB');small=ImageOps.contain(image,(512,512),Image.Resampling.BILINEAR)
    gray=np.asarray(small.convert('L'));mask=ndimage.binary_closing(gray<100,structure=np.ones((3,3)))
    labelled,count=ndimage.label(mask);boxes=ndimage.find_objects(labelled);areas=np.bincount(labelled.ravel());candidates=[];w,h=small.size
    for index,slices in enumerate(boxes,1):
        if slices is None:continue
        ys,xs=slices;bw,bh=xs.stop-xs.start,ys.stop-ys.start;cx=(xs.start+xs.stop)/(2*w);cy=(ys.start+ys.stop)/(2*h)
        if areas[index]<20 or not(.02<=bw/w<=.35 and .04<=bh/h<=.55 and 1.2<=bh/max(bw,1)<=8):continue
        if not(.25<=cx<=.75 and .15<=cy<=.7 and ys.stop/h<=.8):continue
        candidates.append((float(areas[index])*(1-abs(cx-.5)),(xs.start,ys.start,xs.stop,ys.stop)))
    if candidates:
        _,box=max(candidates);x1,y1,x2,y2=box;mx=(x2-x1)*.2;my=(y2-y1)*.2
        source_box=(max(0,int((x1-mx)*image.width/w)),max(0,int((y1-my)*image.height/h)),
            min(image.width,int(np.ceil((x2+mx)*image.width/w))),min(image.height,int(np.ceil((y2+my)*image.height/h))))
        crop=image.crop(source_box)
    else:source_box=(0,0,image.width,image.height);crop=image
    prepared=ImageOps.pad(crop,(224,224),method=Image.Resampling.LANCZOS,color=(128,128,128))
    return prepared,{'version':'body-center-component-v1','proposal_found':bool(candidates),'source_box':list(source_box),
        'scope':'unlabelled intensity/shape proposal, not a validated defect localization result'}
