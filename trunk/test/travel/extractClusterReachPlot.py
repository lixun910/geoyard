import numpy as np
import scipy
import sys
import operator
import random
import pylab as P
from ShapeHelper import *

    
class Node():
    def __init__(self,start,end,count):
        self.start = start
        self.end = end
        self.count = count
        self.id = 0
        self.children = []
        self.parent = None
        self.leaf_nodes = None
        
    def addChild(self,node):
        self.children.append(node)
        
    def hasChild(self):
        return len(self.children) > 0
    
    def getHierarchicalLevel(self):
        level = 1
        sub_level = 0
        for node in self.children:
            sub_level = max(sub_level, node.getHierarchicalLevel())
        return sub_level + level        
        
    def _getAllLeafNodes(self):
        leaf_nodes = []
        if not self.hasChild():
            leaf_nodes = [self]
        
        for node in self.children:
            leaf_nodes += node._getAllLeafNodes()
            
        return leaf_nodes
    
    def getAllLeafNodesNumber(self):
        if self.leaf_nodes == None:
            self.leaf_nodes = self._getAllLeafNodes()
        return len(self.leaf_nodes)
        
        
def exportCluster(points,order, clusters,shapefilename):
    shapeType = shapelib.SHPT_POINT
    shapeFile = shapelib.create(shapefilename, shapeType)
    dbfName = shapefilename[:-3] + 'dbf'
    dbf = dbflib.create(dbfName)
    dbf.add_field('ID', dbflib.FTInteger, 50,0)
    dbf.add_field('Cluster', dbflib.FTInteger, 50,0)
   
    for i,id in enumerate(order):        
        p = list(points[id])
        p.append(0)
        shapeObject = [tuple(p)]
        obj = shapelib.SHPObject(shapeType, -1, [shapeObject])
        shapeFile.write_object(-1, obj)
        cluster_id = -1
        for cluster in clusters:
            if cluster.start <= i <= cluster.end:
                cluster_id = cluster.id
                break
        dbf.write_record(i, {'ID':id,'Cluster':cluster_id})
        
    shapeFile.close()
    dbf.close()
    
def createConvexHull(points,order, clusters,shapefilename):
    m = len(clusters)
    point_sets = [[] for i in range(m+1)]
    
    for i,id in enumerate(order):        
        p = list(points[id])
        p.append(.0)
        cluster_id = -1
        for cluster in clusters:
            if cluster.start <= i <= cluster.end:
                cluster_id = cluster.id
                break
        point_sets[cluster_id].append(p)
    
    convex_hull_set = []
    from shapely.geometry import MultiPoint, asMultiPoint
    for i,point_set in enumerate(point_sets):
        mp = asMultiPoint(point_set)
        convex_hull_set.append(mp.convex_hull)
        
    shapeType = shapelib.SHPT_POLYGONZ
    shapeFile = shapelib.create(shapefilename, shapeType)
    dbfName = shapefilename[:-3]+ 'dbf'
    dbf = dbflib.create(dbfName)
    dbf.add_field('ID', dbflib.FTInteger, 50,0)
    dbf.add_field('Cluster', dbflib.FTInteger, 50,0)
        
    for i,convex_hull in enumerate(convex_hull_set):       
        cluster_id = i
        if cluster_id == m: cluster_id = -1
        shapeObject = np.array(convex_hull.exterior)
        n = len(shapeObject)
        shapeObject = np.append(shapeObject,[[.0] for j in range(n)],axis=1)
        shapeObject = [tuple(j) for j in shapeObject]
        obj = shapelib.SHPObject(shapeType, -1, [shapeObject])
        shapeFile.write_object(-1, obj)
        dbf.write_record(i, {'ID':id,'Cluster':cluster_id})
        
    shapeFile.close()
    dbf.close()

def extractCluster(RD,t,minPts,resultsFileName=''):
    print np.array(RD)
    RD[0] = 0.35
    m = len(RD)
    
    SDAset = []
    MIBset = []
    ClusterSet = []
    index = 0
    while index < m-1:
        #if start of down area D:
        if RD[index]*(1-t) >= RD[index+1]:
            #add D to steep down areas
            start = index
            SDA = []
            SDA.append(index)
            index += 1
            end = index
            while index<m-1 and RD[index] >=RD[index+1]:
                SDA.append(index)
                #if RD[index]*(1-t) >= RD[index+1]:
                #    end = index - start +1
                end = index -start + 1
                index += 1
            SDA.append(index)
            SDA = SDA[:end+1]
            #if 0< len(SDA) <= minPts:
            if len(SDA) >= 2:
                SDAset.append(SDA)
                try:
                    currentMIB = RD[SDA[0]] #MIBset[-1]
                    MIBset = [max(i,currentMIB) for i in MIBset]
                except:
                    pass
                MIBset.append(0)
            #index = end of D
            
        #elif start of steep up area U:
        elif RD[index] <= RD[index+1]*(1-t):
            #index = end of U
            start = index
            SUA = []
            SUA.append(index)
            index += 1
            end = index
            count = 0
            while index < m-1 and RD[index+1] >= RD[index]:
                SUA.append(index)
                if RD[index] <= RD[index+1]*(1-t):
                    end = index -start + 1
                index += 1
                count+=1
                
            SUA.append(index)
            SUA = SUA[:end+1]
            #if 0< len(SUA) <= minPts:
            if len(SUA) >= 2:
                #for each steep down area D:
                numSDA = len(SDAset)
                # reorganize MIBset
                
                for i,SDA in enumerate(reversed(SDAset)):
                    #if D and U form a cluster:
                    start = -1
                    end = -1
                    if MIBset[numSDA-i-1] >= RD[SUA[-1]]:
                        continue
                    if SUA[-1] - SDA[0] > minPts:
                        #if RD[SDA[0]]*(1-t) >= RD[SUA[-1]]:
                        if RD[SDA[0]] > RD[SUA[-1]] > RD[SDA[-1]]:
                            end = SUA[-2]
                            start = -1
                            for j in range(len(SDA)):
                                if RD[SDA[j]] <= RD[SUA[-1]]:
                                    start = SDA[j] -1
                                    break
                            if MIBset[numSDA-i-1] >= RD[end]:
                                continue
                        #elif RD[SDA[0]] <= RD[SUA[-1]]*(1-t):
                        elif RD[SUA[0]]< RD[SDA[0]] <= RD[SUA[-1]]:
                            start = SDA[0]
                            end = SUA[-2]
                            for j in range(len(SUA)):
                                if RD[SUA[j]] >= RD[start]:
                                    if end > SUA[j]:
                                        end = SUA[j]
                                    break
                            if MIBset[numSDA-i-1] >= RD[start]:
                                continue
                        #add [start(D), end(U)] to set of clusters
                        if start >= 0 and end >0 and end - start >= minPts:
                            ClusterSet.append([start,end-1])
                currentMIB = max(RD[index], MIBset[-1])
                MIBset = [max(i,currentMIB) for i in MIBset] 
        else:
            index += 1
    
    if len(resultsFileName) > 0:
        f = open(resultsFileName,'w')
        for cl in ClusterSet:
            f.write('%s%s%s\n' % (cl , '\t',cl[1] - cl[0]))
        f.close()
    return ClusterSet

def extractRangeCollection(filename):
    starts= []
    ends = []
    counts = []
    f = open(filename)
    line = f.readline()
    while(len(line)>0):
        line = line.strip().split('\t')
        count = int(line[1])
        info = line[0].strip()[1:-1]
        info = info.split(',')
        start,end = int(info[0]),int(info[1])
        counts.append(count)
        starts.append(start)
        ends.append(end)
        line = f.readline()
    f.close()
    
    return np.array(starts),np.array(ends),np.array(counts)

def extractHierarchicalStructure(starts,ends,counts,root):
    m = len(starts)
    if m == 0: return
    
    # find the largest cluster
    max_pos = counts.argmax()
    max_start,max_end = starts[max_pos],ends[max_pos]
    node = Node(max_start,max_end,counts[max_pos])
    
    # find and process its descendent clusters
    desc_start = np.where(starts[:max_pos]>=max_start)
    if len(desc_start[0]) > 0: 
        desc_start = desc_start[0][0]
        extractHierarchicalStructure(starts[desc_start:max_pos],ends[desc_start:max_pos],counts[desc_start:max_pos],node)
    root.addChild(node)
    
    if isinstance(desc_start,int) and desc_start > 0:
        # find and process left clusters
        leftStarts,leftEnds, leftCounts = starts[:desc_start],ends[:desc_start],counts[:desc_start]
        i = len(leftStarts) -1
        while i >=0:
            start,end = leftStarts[i],leftEnds[i]
            node = Node(start,end,leftCounts[i])
            idxs = np.where(leftStarts[:i]>=start)
            if len(idxs[0]) == 0:
                root.addChild(node)
                i -= 1
            else:
                idx = idxs[0][0]
                sub_starts,sub_ends,sub_counts = leftStarts[idx:i],leftEnds[idx:i],leftCounts[idx:i]
                extractHierarchicalStructure(sub_starts,sub_ends,sub_counts,node)
                root.addChild(node)
                i = idx-1
    
    if max_pos < m - 1:
        # find and process right clusters
        leftStarts,leftEnds, leftCounts = starts[max_pos+1:],ends[max_pos+1:],counts[max_pos+1:]
        i = len(leftStarts) -1
        while i >=0:
            start,end = leftStarts[i],leftEnds[i]
            node = Node(start,end,leftCounts[i])
            idxs = np.where(leftStarts[:i]>=start)
            if len(idxs[0]) == 0:
                root.addChild(node)
                i -= 1
            else:
                idx = idxs[0][0]
                sub_starts,sub_ends,sub_counts = leftStarts[idx:i],leftEnds[idx:i],leftCounts[idx:i]
                extractHierarchicalStructure(sub_starts,sub_ends,sub_counts,node)
                root.addChild(node)
                i = idx-1
                
def reorganizeHCluster(root):
    m = len(root.children)
    if m > 0:
        if m == 1:
            # merge its children to this node
            childNode = reorganizeHCluster(root.children[0])
            root.children = childNode.children
        else:
            # no need to merge, need to reorganize its children
            children = root.children
            root.children = []
            for child in children:
                root.children.append(reorganizeHCluster(child))
    return root


def getClusterByHierarchy(root,hierarch):
    if hierarch == 0: return []

    clusters = [root]
    while hierarch >0:
        removedClusters = []
        newClusters = []
        for cluster in clusters:
            if len(cluster.children) > 0:
                newClusters += cluster.children 
                removedClusters.append(cluster)
        
        for cluster in removedClusters:
            clusters.remove(cluster)
        for cluster in newClusters:
            clusters.append(cluster)
            
        hierarch -= 1
        
    return clusters

color_set = 'grcmyk' #6
def drawHCluster(root,x,y,start,level,color=None):
    if root.start==root.end==0:
        P.text(x-10,y+0.1,'Photo Hierarchical Clustering Root')        
    
    if y / 2 == level+1:
        color = color_set[random.randint(0,5)]
        
    if color == None:
        line_color = 'b-'
        point_color = 'bo'
        font_color = 'b'
    else:
        line_color = color + '-'
        point_color = color + 'o'
        font_color = color
    
    if root.hasChild():
        _y = y - 2
        P.plot([x,x],[y,_y], line_color)
        children = root.children
        children.sort(key=operator.attrgetter('start'))  
        children_x = []
        children_y = []
        
        for node in children:
            leafnodes_number = node.getAllLeafNodesNumber()
            _x = start + leafnodes_number / 2.0
            children_x.append(_x)
            children_y.append(_y)
            if not node.hasChild():
                _line_color = line_color
                _point_color = point_color
                _font_color = font_color
                if y / 2 > level+1:
                    _color = color_set[random.randint(0,5)]
                    _line_color = _color + '-'
                    _point_color = _color + 'o'
                    _font_color = _color
                P.plot([_x,_x],[_y,0],_line_color) 
                P.plot(_x,0,_point_color)
                P.text(_x,-1.5,str(node.start)+'-'+str(node.end),color=_font_color,rotation=90)
            else:
                drawHCluster(node,_x,_y,start,level,color)
            start += leafnodes_number
            P.plot(children_x,children_y,line_color)   
    
    
        
def Test1():
    cluster_result_filename = 'cluster_test.txt'
    
    from optics import *
    from numpy import random
    import scipy as S
    testX = random.rand(100, 2)
    x1 = S.rand(30,2)*2
    x2 = (S.rand(40,2)+1)*2
    x3 = (S.rand(40,2)*0.2+1)*2.5
    testX= np.concatenate((x1,x2,x3))
    
    #P.plot(testX[:,0], testX[:,1], 'ro')
    #RD, CD, order = optics(testX, 4)
    k =4 
    Eps = epsilon(testX,k)
    print Eps
    RD,CD,order = loptics(testX, Eps,k) 
    testXOrdered = testX[order]
   
    P.plot(testXOrdered[:,0], testXOrdered[:,1], 'b-')
    x = [float(i)/10.0 for i in range(110)]
    y = [-RD[i] for i in order]
    P.vlines(x,[0],y)
    print order
    P.savefig('D:\\Desktop\\test.png',dpi=300,format='png')

    ##############################################################
    _RD = [RD[i] for i in order]
    clusters = extractCluster(_RD,0.1,k,cluster_result_filename)
    
    hierarchy = 1
    starts,ends,counts = extractRangeCollection(cluster_result_filename)
    root = Node(0,0,0)
    extractHierarchicalStructure(starts,ends,counts,root)
    clusters = getClusterByHierarchy(root,hierarchy)
    
    for i,cluster in enumerate(clusters):
        cluster.id = i
        print cluster.start, ',', cluster.end, ',', cluster.count
    print "##############################"
        
    #################################################################
    color = ['ro','bo','yo','go','ko','mo','co','r*','b*']
    for i,id in enumerate(order):
        cluster_id = -1
        for cluster in clusters:
            if cluster.start <= i <= cluster.end:
                cluster_id = cluster.id
                break
        P.plot(testX[id,0],testX[id,1],color[cluster_id])
    P.show()
        
def Test(filter):
    source_filename = sys.argv[1]
    reach_plot_result_filename = sys.argv[2]
    cluster_result_filename = sys.argv[3]
    start_hierarchy = int(sys.argv[4])
    end_hierarchy = int(sys.argv[5])
    
    shape_type, xy = read_objects(source_filename)
    xy = scipy.array(xy)
    points = np.column_stack((xy[:,0],xy[:,1]))
    
    #########################################
    f = open(reach_plot_result_filename)
    line1 = f.readline().strip()
    line2 = f.readline().strip()
    f.close()
    
    _RD = [float(i) for i in line1.split(',')]
    order = [int(i) for i in line2.split(',')]
    RD = [_RD[i] for i in order]
    clusters = extractCluster(RD,filter,150,cluster_result_filename)
    #save to cluster.txt
    
    ###########################################
    for hierarchy in range(start_hierarchy,end_hierarchy):
        starts,ends,counts = extractRangeCollection(cluster_result_filename)
        root = Node(0,0,0)
        extractHierarchicalStructure(starts,ends,counts,root)
        root = reorganizeHCluster(root)
        
        #################################################
        """
        hierarchy_height = 2
        min_node_interval = 1
        total_hierarchical_number = root.getHierarchicalLevel()
        total_leafnodes_number = root.getAllLeafNodesNumber()
        graph_height = total_hierarchical_number * hierarchy_height
        graph_width = total_leafnodes_number * min_node_interval
        x,y = (graph_width / 2.0, graph_height )
        drawHCluster(root,x,y,0,17)
        
        P.axhline(y=17*2, color='r',linestyle='--')
        P.text(-1,-0.5,'Pointset Range:',color='r',rotation=90)       
        P.show()
        """
        
        #######################################################
        clusters = getClusterByHierarchy(root,hierarchy)        
        for i,cluster in enumerate(clusters):
            cluster.id = i
            print cluster.start, ',', cluster.end, ',', cluster.count
            
        exportCluster(points,order, clusters,cluster_result_filename[:-3]+str(hierarchy)+'.shp')
        createConvexHull(points,order, clusters,cluster_result_filename[:-3]+str(hierarchy)+'.convexhull.shp')
    
    ###########################################
    
def generate_dendrogram(root):
    from hcluster import pdist, linkage, dendrogram
    import numpy
    from numpy.random import rand
    import matplotlib
    
    X = rand(10,100)
    X[0:5,:] *= 2
    Y = pdist(X)
    Z = linkage(Y)
    print Y
    print Z
    dendrogram(Z)
    
if __name__ == "__main__":
    #generate_dendrogram(None)
    #Test1()
    Test(0.001) 
    