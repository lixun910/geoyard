import sys
import pylab as P
import scipy as S
import numpy as N
from optics import *
from ShapeHelper import *
import shapelib
import time
import pickle
from datetime import datetime, date, time
from shapely.geometry import *


def GetCAPoints():
    o = open('CA.csv','w')
    f = open('xy.csv')
    line = f.readline()
    print line
    i=0
    while len(line)>0:
        line = line.strip()
        line = line.split(' ')
        y = float(line[0])
        x = float(line[1])
        if x > -124.74 and x < -113.99 and y >32.417 and y < 42.04:
            o.write('%s %s\n' % (y,x))
            i += 1
        line = f.readline()
    f.close()
    o.close()
    
def GetAllPoints():
    o = open('All.csv','w')
    f = open('xy.csv')
    line = f.readline()
    print line
    i=0
    while len(line)>0:
        line = line.strip()
        line = line.split(' ')
        y = float(line[0])
        x = float(line[1])
        if y > -70 and y < 70 and x >-60 and x < 60:
            o.write('%s %s\n' % (y,x))
            i += 1
        line = f.readline()
    f.close()
    o.close()

def pointsFromFile(filename):
    xy = []
    f = open(filename)
    line = f.readline()
    while len(line) > 0:
        line = line.strip()
        line = line.split(' ')
        y = float(line[0])
        x = float(line[1])
        xy.append([(x,y,0)])
        #xy.append([x,y])
        line = f.readline()
    f.close()
    #return S.array(xy)
    return xy
    
def export2Shape():
    xy= pointsFromFile('CA.csv')
    create_shapefile(shapelib.SHPT_POINT, xy, 'CA.shp')
                    
def exportOpticsCluster(points, RD,CD,order,threshold,shapefilename):
    shapeType = shapelib.SHPT_POINT
    shapeFile = shapelib.create(shapefilename, shapeType)
    dbfName = shapefilename[:-3] + 'dbf'
    dbf = dbflib.create(dbfName)
    dbf.add_field('ID', dbflib.FTInteger, 50,0)
    dbf.add_field('RD', dbflib.FTDouble, 50,10)
    dbf.add_field('CD', dbflib.FTDouble, 50,10)
    dbf.add_field('Order', dbflib.FTInteger, 50,0)
    dbf.add_field('Cluster', dbflib.FTInteger, 50,0)
   
    noise = False 
    noise = 0
    clusterid = 1
    cluster = noise 
    
    cluster = 1
    for i,id in enumerate(order):        
        p = points[id]
        p = list(p)
        p.append(0)
        shapeObject = [tuple(p)]
        obj = shapelib.SHPObject(shapeType, -1, [shapeObject])
        shapeFile.write_object(-1, obj)
        
        if RD[id] >= threshold and noise == False:
            cluster+=1
            noise = True
            dbf.write_record(i, {'ID':id,'RD':RD[id],'CD':CD[id],'Order':i,'Cluster':0})
            continue
        elif RD[id] >= threshold and noise == True:
            dbf.write_record(i, {'ID':id,'RD':RD[id],'CD':CD[id],'Order':i,'Cluster':0})
            continue
        
        dbf.write_record(i, {'ID':id,'RD':RD[id],'CD':CD[id],'Order':i,'Cluster':cluster})
        noise = False
        """
        if RD[id] > threshold:
            if CD[id] <= threshold:
                clusterid += 1
                cluster = clusterid 
            else:
                cluster = noise                
            dbf.write_record(i, {'ID':id,'RD':RD[id],'CD':CD[id],'Order':i,'Cluster':cluster})
        else:
            dbf.write_record(i, {'ID':id,'RD':RD[id],'CD':CD[id],'Order':i,'Cluster':cluster})
        """ 
    shapeFile.close()
    dbf.close()
    
if __name__ == "__main__":
    print '##############################################'
    #GetAllPoints()
    #GetCAPoints()
    #export2Shape()
   
    sourcePointFile = sys.argv[1]
    shape_type, xy = read_objects(sourcePointFile)
    xy = S.array(xy)
    #P.scatter(xy[:,0],xy[:,1],alpha =  0.5)
   
    # cluster
    xy = N.column_stack((xy[:,0],xy[:,1]))
    MinPts = int(sys.argv[2])
    thres = float(sys.argv[3])
    Eps = epsilon(xy,MinPts)

    print time.ctime()
    RD,CD,order = loptics(xy,Eps,MinPts)
    print time.ctime()
    #exportOpticsCluster(xy, RD, CD, order, thres, 'CAcluster.shp')
    
    # write RD,Order to file
    f1 = open('ReachPlot_' + sys.argv[2] + '.txt','w')
    f1.write('%s\n'%(','.join(str(i) for i in RD)))
    f1.write('%s\n'%(','.join(str(i) for i in order)))
    f1.close()
    
    # display for OPTICS
    xlabel = 'eps=' + str(Eps) + ',MinPts=' + str(MinPts) +'            Cluster-order of the Photos' 
    P.xlabel(xlabel)
    P.ylabel('Reachability Distance')
    P.title('Reachability Plot for Geo-tagged Photos with a Hierachical Clusters')
    _x = [float(i) for i in range(len(xy))]
    _y = [RD[i] for i in order]
    P.axhline(y=thres, color='r')
    P.vlines(_x,[0],_y)
    P.savefig('ReachPlot_'+sys.argv[2]+'.png',dpi=300,format='png')
    P.show()
