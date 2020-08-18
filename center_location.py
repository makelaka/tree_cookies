import numpy as np
import scipy.ndimage as nd

#Find center of circular tree SLICE
def find_center(img):
    #accessory functions
    def rowLabels(w,h):
        return np.tile(range(h),(w,1))
    def colLabels(w,h):
        return rowLabels(h,w).T
    def diagSum(a,d=1):
        w,h = a.shape
        l = colLabels(w,h)*d+rowLabels(w,h)
        return nd.sum(a,l,list(range(np.amin(l),np.amax(l)+1)))
    def diagMean(a,d=1,mask=None):
        if mask is None:
            mask = np.ones_like(a)
        num = diagSum(mask,d)
        return np.divide(diagSum(a,d),num,where=num!=0)
    def shear(a,d = 1):
        rows, cols = a.shape
        if cols > rows:
            a = a.T
            rows, cols = a.shape
        stacked = np.vstack((a, a))
        major_stride, minor_stride = stacked.strides
        strides = major_stride, minor_stride * (cols + d)
        shape = (rows, cols)
        return np.lib.stride_tricks.as_strided(stacked, shape, strides)
    #filter definitions
    gx = np.array([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]])
    gy = np.transpose(gx)
    r2 = .5**.5
    sxy = gx*r2+gy*r2
    syx = gx*r2-gy*r2
    w,h = img.shape
    #gaussian blur
    blur = nd.gaussian_filter(img,sigma=1)
    #set up sobel filters
    dx  = nd.convolve(blur,weights=gx)
    dy  = nd.convolve(blur,weights=gy)
    dxy = nd.convolve(blur,weights=sxy)
    dyx = nd.convolve(blur,weights=syx)
    #sobel filters
    sob1 = np.hypot(dx,dy)
    sob2 = np.hypot(dx,-dy)
    sob3 = np.hypot(dxy,dyx)
    sob4 = np.hypot(dxy,-dyx)
    m1 = np.abs(sob1).astype(img.dtype)
    m2 = np.abs(sob2).astype(img.dtype)
    m3 = np.abs(sob3).astype(img.dtype)
    m4 = np.abs(sob4).astype(img.dtype)
    #blurred regions
    bm1 = nd.gaussian_filter(m1,sigma=10)
    bm2 = nd.gaussian_filter(m2,sigma=10)
    bm3 = nd.gaussian_filter(m3,sigma=10)
    bm4 = nd.gaussian_filter(m4,sigma=10)
    #intensity along line through middle of region
    a1 = diagMean(bm1,-1)
    a2 = diagMean(bm2)
    a3 = bm3.mean(1,keepdims=False)
    a4 = bm4.mean(0,keepdims=False)
    #replace pixels with average intensity along line
    aa1 = np.tile(a1,(w+h-2,1))
    aa1 = np.reshape(aa1,(-1,w+h-2))[-w:,:h]
    aa2 = np.tile(a2,(w+h,1))
    aa2 = np.reshape(aa2,(-1,w+h))[:w,:h]
    aa3 = np.tile(a3,(h,1)).T
    aa4 = np.tile(a4,(w,1))
    #sum up images
    a12 = aa1*aa2**2
    a34 = aa3*aa4**2
    a1234 = a12+a34
    a1234 = nd.gaussian_filter(a1234,sigma=25)
    #center
    p = nd.minimum_position(a1234)
    return p
