import numpy as np
from skspatial.objects import Points, Line

#z parameters remove partial slices from ends of stack
#z1 = first full slice, default 50
#z2 = n - last full slice, default 50
def center_regression(in_file,z1=50,z2=50,out_file="centers.txt"):
    file = open(in_file, "r") #original centers
    centers = []
    z = 0
    for line in file:
        center = line.strip().strip("(").strip(")").split(",")
        center = [int(center[0]),int(center[1]),z]
        centers.append(center)
        z += 1
    end = z - z2
    new = centers[z1:end] #crop out partial slices
    points = Points(new) #convert data
    line_fit = Line.best_fit(points) #take line of best fit
    p = line_fit.point
    v = line_fit.direction
    p0 = p-p[2]*v/v[2] #initial center
    v0 = v/v[2] #initial direction
    out = open(out_file,"w") #output file
    new_centers = [] #linearized centers
    i = z1 #start with first full slice
    while i <= end: #end with last full slice
        pn = p0+i*v0 #next center
        new_centers.append(pn)
        cn = str((int(pn[0]),int(pn[1]),int(pn[2])))
        out.write(cn) #write center to file
        out.write("\n")
        i += 1
    out.close()
    return new_centers
