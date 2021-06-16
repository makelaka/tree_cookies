import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import operator
import glob

#cost to remove ground truth, cost to remove results, truth, results
def edit_distance(skipCostA = 200, skipCostB = 200, t, r):
	def cost(a,b): #cost to move a point
		return abs(a-b)**2
	def match(A,B,op=operator.add): #match similar points
		costs = dict()
		costs[(-1,-1)] = 0
		trace = dict()
		trace[(-1,-1)] = None
		for i,a in enumerate(A):
			costs[(i,-1)] = op(costs[(i-1,-1)],skipCostA)
			trace[(i,-1)] = (i-1,-1)
		for j,b in enumerate(B):
			costs[(-1,j)] = op(costs[(-1,j-1)],skipCostB)
			trace[(-1,j)] = (-1,j-1)
		for i,a in enumerate(A):
			for j,b in enumerate(B):
				costs[(i,j)],trace[(i,j)] = min(
					(op(costs[(i,j-1)],skipCostB),   (i,j-1)),
					(op(costs[(i-1,j)],skipCostA),   (i-1,j)),
					(op(costs[(i-1,j-1)],cost(a,b)), (i-1,j-1))
				)
		m = (i,j)
		c = costs[m]
		matching = []
		while True:
			mm = trace[m]
			if mm == None:
				break
			if m[0]-mm[0]==1 and m[1]-mm[1]==1:
				matching.append(m)
			m = mm
		matching = matching[::-1]
		return c,matching
	addCost,matching = match(t,r,op=operator.add) #match and score
	#plotting
	segs = [[(t[i],0),(r[j],1)] for i,j in matching]
	fig,ax = plt.subplots(figsize=(10,2))
	lines = LineCollection(segs,colors='.5')
	x = .1
	ax.set_xlim([0, max(t+r)+x])
	ax.set_ylim([0-x, 1+x])
	ax.scatter(t,[0]*len(t))
	ax.scatter(r,[1]*len(r))
	ax.add_collection(lines)
	plt.show()
	return addCost
