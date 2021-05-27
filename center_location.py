import numpy as np #array structures and operations
import scipy.ndimage as nd #gaussian filter, convolve (sobel)
from skspatial.objects import Points, Line #regression of centers
import cv2 #polar conversion

#Find center of 2D tree scan slice
def find_center(img):
    def diagSum(a,d=1): #sum intensity values across a diagonal
        w,h = a.shape
        l = (np.tile(range(h),(w,1)))*d+(np.tile(range(w),(h,1))).T
        return nd.sum(a,l,list(range(np.amin(l),np.amax(l)+1)))
    def diagMean(a,d=1,mask=None): #obtain average intensity value across a diagonal
        if mask is None:
            mask = np.ones_like(a)
        num = diagSum(mask,d)
        return np.divide(diagSum(a,d),num,where=num!=0)
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
    dx  = nd.convolve(blur,weights=gx) #horizontal
    dy  = nd.convolve(blur,weights=gy) #vertical
    dxy = nd.convolve(blur,weights=sxy) #
    dyx = nd.convolve(blur,weights=syx) #
    #step 1. sobel filters
    sob1 = np.hypot(dx,dy)
    sob2 = np.hypot(dx,-dy)
    sob3 = np.hypot(dxy,dyx)
    sob4 = np.hypot(dxy,-dyx)
    #step 2. blur
    bm1 = nd.gaussian_filter(np.abs(sob1).astype(img.dtype),sigma=10)
    bm2 = nd.gaussian_filter(np.abs(sob2).astype(img.dtype),sigma=10)
    bm3 = nd.gaussian_filter(np.abs(sob3).astype(img.dtype),sigma=10)
    bm4 = nd.gaussian_filter(np.abs(sob4).astype(img.dtype),sigma=10)
    #step 3. intensity along line through middle of region
    a1 = diagMean(bm1,-1)
    a2 = diagMean(bm2)
    a3 = bm3.mean(1,keepdims=False)
    a4 = bm4.mean(0,keepdims=False)
    #step 4. replace pixels with average intensity along line
    aa1 = np.tile(a1,(w+h-2,1))
    aa1 = np.reshape(aa1,(-1,w+h-2))[-w:,:h]
    aa2 = np.tile(a2,(w+h,1))
    aa2 = np.reshape(aa2,(-1,w+h))[:w,:h]
    aa3 = np.tile(a3,(h,1)).T
    aa4 = np.tile(a4,(w,1))
    #step 5. sum up images
    a12 = aa1*aa2**2
    a34 = aa3*aa4**2
    a1234 = a12+a34
    a1234 = nd.gaussian_filter(a1234,sigma=25)
    p = nd.minimum_position(a1234) #center
    return p

#compute regression of individual centers to line up slices
#z parameters remove partial slices from ends of stack
#z1 = first full slice, default 50
#z2 = n - last full slice, default 50
def regression(centers=[]):
    points = Points(centers) #convert data
    line_fit = Line.best_fit(points) #take line of best fit
    p = line_fit.point
    v = line_fit.direction
    p0 = p-p[2]*v/v[2] #initial center
    v0 = v/v[2] #initial direction
    new_centers = [] #linearized centers
    i = z1 #start with first full slice
    while i <= end: #end with last full slice
        pn = p0+i*v0 #next center
        new_centers.append(pn)
        i += 1
    return new_centers

#convert slices to polar coordinates
def polar_conversion(data,centers):
    data = data.astype(np.float32) #make sure data is in correct form
    polars = []
    i = 0
    for slice in data:
        value = np.sqrt(((centers[i][0])**2.0)+((centers[i][1])**2.0))
        polar_image = cv2.linearPolar(img,(centers[i][0], centers[i][1]), value, cv2.WARP_FILL_OUTLIERS)
        polars.append(polar_image)
        i += 1
    return polars
