from tifffile import imread,imshow,imwrite #for the data
import numpy as np #data manipulation
import scipy.ndimage as nd #gaussian blur
#import cv2 #uncomment to convert from polar to original coordinates at the end

def edge_detection(polars,threshold=0.0,wrap=10,blur=2):
    #ADD PADDING?
    polars[polars<threshold*np.max(polars)] = 0 #threshold out weak intensities
    z,h,w = polars.shape #image dimensions
    unpadded_shape = (z,h-20,w) #dimensions without wrap
    rpolar = nd.gaussian_filter(polars,sigma=blur)[:,wrap:-wrap,:] #blur
    #ref = np.max(rpolar)-rpolar
    ref = rpolar
    stack = []
    weights = np.ones_like(rpolar) #to accumulate persistence values
    prefixWeights = np.cumsum(weights,axis=2)
    # leftmost point of superlevelset:
    #  l(x) = min{ l | 0<=l<=x and (f(m),m)>=(f(x),x) for all l<=m<x }
    def pairingLeft(z,y,x):
        global stack
        fx = ref[z,y,x]
        if x==0:
            stack = [(0,0)]
        if x==0 or ref[z,y,x-1]<=fx: #increasing
            l = x
        else:
            l = 0
            while stack:
                l,fl = stack[-1]
                stack = stack[:-1]
                if fl>=fx and (l==0 or ref[z,y,l-1]<=fx):
                    stack.append((l,fx))
                    break
        stack.append((x,fx))
        return l
    # rightmost point of superlevelset:
    #  r(x) = max{ r | x<=r<w  and (f(x),x)<=(f(m),m) for all x<m<=r }
    def pairingRight(z,y,x):
        global stack
        fx = ref[z,y,x]
        if x==0:
            stack = [(0,0)]
        if x==0 or ref[z,y,x-1]<fx: # increasing
            l = x
        else:
            l = 0
            while stack:
                l,fl = stack[-1]
                stack = stack[:-1]
                if fl>=fx and (l==0 or ref[z,y,l-1]<fx):
                    stack.append((l,fx))
                    break
        stack.append((x,fx))
        return l
    leftRay = np.fromfunction(np.vectorize(pairingLeft),ref.shape,dtype=ref.dtype)
    ref = np.flip(ref,axis=2)
    rightRay = np.fromfunction(np.vectorize(pairingRight),ref.shape,dtype=ref.dtype)
    rightRay = np.flip(rightRay,axis=2)
    rightRay = w-1-rightRay
    prefixSum = np.cumsum(rpolar*weights,axis=2)
    # calculate area to the left or right of a possible edge
    def areaLeft(z,y,x):
        x0 = leftRay[z,y,x]
        return (prefixSum[z,y,x]-prefixSum[z,y,x0])-rpolar[z,y,x]*(prefixWeights[z,y,x]-prefixWeights[z,y,x0])
    def areaRight(z,y,x):
        x1 = rightRay[z,y,x]
        return (prefixSum[z,y,x1]-prefixSum[z,y,x])-rpolar[z,y,x]*(prefixWeights[z,y,x1]-prefixWeights[z,y,x])
    aLeft  = np.fromfunction(areaLeft,ref.shape,dtype=ref.dtype)
    aRight = np.fromfunction(areaRight,ref.shape,dtype=ref.dtype)
    aLeft  = -aLeft
    aRight = -aRight
    aPers = np.minimum(aLeft,aRight,dtype=padded.dtype)
    imgOut = aPers**.1 #reduce intensity range
    #uncomment below to threshold again - probably not necessary
    #imgOut[imgOut<threshold*np.max(imgOut)] = 0
    #uncomment below to unwrap image back to circular tree rings
    #aPersUnpolar = np.stack([cv2.linearPolar(aPers[i],ps[i],r,cv2.WARP_INVERSE_MAP) for i in range(z)])
    return imgOut
