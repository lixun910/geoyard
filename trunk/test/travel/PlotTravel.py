from mpl_toolkits.basemap import Basemap
import pylab
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Wedge, Polygon,Rectangle
from matplotlib.collections import PatchCollection
import numpy as np
import string,copy,pickle
import shapely.geometry as shapely
from ShapeHelper import *
from TravelMining import *

def gen_centroid(point):
    c = []
    #c.append(Circle(point, 15000,ec="none",color='g'))
    #c.append(Circle(point, 5000,ec="none",color='w'))
    # SF LA
    #c.append(Circle(point, 500,ec="none",color='g'))
    #c.append(Circle(point, 1500,ec="none",color='w'))
    # SF downtown
    c.append(Circle(point, 100,ec="none",color='g'))
    c.append(Circle(point, 400,ec="none",color='w'))
    
    return c

def gen_travel_flow(_start, _end, weight=1):
    start = np.array(copy.deepcopy(_start))
    end = np.array(copy.deepcopy(_end))
    flow_arrow = []
    
    length = np.linalg.norm(end - start)
    cos_theta = (end - start)[0] / length
    sin_theta = (end - start)[1] / length
    
    #offset_x = 25000 * cos_theta
    #offset_y = 25000 * sin_theta
    #LA
    #offset_x = 2500 * cos_theta
    #offset_y = 2500 * sin_theta
    # SF downtown
    offset_x = 500 * cos_theta
    offset_y = 500 * sin_theta
    
    start += [offset_x, offset_y]
    end -= [offset_x,offset_y]
    
    #offset = 2000
    #thick = weight * 100000
    #LA
    #offset = 200
    #thick = weight * 30000
    # SF downtown
    offset = 40
    thick = weight * 3000
    length = np.linalg.norm(end - start)
    
    flow_arrow.append([0,thick+offset])
    flow_arrow.append([(length-2*thick),thick+offset])
    flow_arrow.append([(length-2*thick),2*thick+offset])
    flow_arrow.append([length,0+offset])
    flow_arrow.append([0,0+offset])    
    flow_arrow.append([0,thick+offset])    
    flow_arrow = np.array(flow_arrow)

    # move and rotation        
    cos_theta = (end - start)[0] / length
    sin_theta = (end - start)[1] / length
    
    for i,p in enumerate(flow_arrow):
        x = p[0]*cos_theta - p[1]*sin_theta
        y = p[0]*sin_theta + p[1]*cos_theta
        flow_arrow[i] = np.array([x,y])
    flow_arrow += start
    return matplotlib.patches.Polygon(flow_arrow,True,color='g')
   
def project_polygon(m,polygons):
    for i,polygon in enumerate(polygons):
        for j,p in enumerate(polygon):
            polygons[i][j] = m(p[0],p[1])
    return polygons
    
def project_points(m,points):
    for i,p in enumerate(points):
        points[i] = m(p[0],p[1])
    return points
        
def plot_travel_pattern(source_filename,flows):
    centroid_patches =[]
    flow_patches = []
    region_patches = []
    background_pathches = []

    m = Basemap(llcrnrlon=-124.74,llcrnrlat=32.417,urcrnrlon=-113.99,urcrnrlat=42.04,lat_ts=20, resolution='h',projection='merc')
    
    #################################################################
    ## background
    background_filename = 'Data\\co06_d00.shp'
    shape_type, polygons = read_objects(background_filename)
    polygons = project_polygon(m,polygons)
    for polygon in polygons:
        background_pathches.append(matplotlib.patches.Polygon(np.array(polygon),True))
    
    #################################################################
    ## regions and centroids
    centroid = []
    shape_type, polygons = read_objects(source_filename)
    polygons = project_polygon(m,polygons)
    for polygon in polygons:
        center = shapely.Polygon(polygon).centroid
        centroid.append([center.x,center.y])
        region_patches.append(matplotlib.patches.Polygon(np.array(polygon),True))
    #demo = np.array([[0,0+100000],[100000,0+100000],[100000,50000+100000],[60000,70000+100000],[0,30000+100000]])
    #region_patches.append(matplotlib.patches.Polygon(demo,True))
    for i in range(len(centroid)):
        centroid_patches += gen_centroid(centroid[i])
    centroid_patches += gen_centroid([0,0])
        
    
    #################################################################
    ## flows
    for flow in flows:
        flow = gen_travel_flow(centroid[flow[0]],centroid[flow[1]],flows[flow])
        flow_patches.append(flow)
    #flow_patches.append(gen_travel_flow([200000,-10000],[0,-10000],0.1))
    #################################################################
    ## points
    source_filename = 'Data\\CA.shp' #sys.argv[1]
    shape_type, xy = read_objects(source_filename)
    xy = np.array(xy)
    #points = np.column_stack((xy[:,0],xy[:,1]))
    x = xy[:,0]#longitudes
    y = xy[:,1]#latitudes
    x,y=m(x,y)
    
    
    #################################################################
    ## Plotting
    fig=pylab.figure(figsize=(10, 10), dpi=80,)
    #ax=fig.add_subplot(111,frameon=False, xticks=[], yticks=[],axis_bgcolor='white')
    ax=fig.add_subplot(111)
    # points
    plt.scatter(x,y,s=1,alpha =  0.5)
    # background
    collection = PatchCollection(background_pathches, cmap=matplotlib.cm.jet, alpha=0.3)
    collection.set_color('gray')
    ax.add_collection(collection)
    # regions
    colors = 100*np.random.rand(len(region_patches))
    collection = PatchCollection(region_patches, cmap=matplotlib.cm.jet, alpha=0.3)
    collection.set_array(np.array(colors))
    ax.add_collection(collection)
    # centroids
    collection = PatchCollection(centroid_patches, cmap=matplotlib.cm.jet, alpha=0.4)
    ax.add_collection(collection)
    """
    for i in range(len(centroid)):
        plt.text(centroid[i][0] + 18000,centroid[i][1],chr(65+i),family="san-serif",size=12)
        ann = ax.annotate(chr(65+i), xy=(centroid[i][0] + 18000,centroid[i][1]),  xycoords='data',
                xytext=(35, 0), textcoords='offset points',
                size=20, va="center",
                bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec="none"),
                arrowprops=dict(arrowstyle="wedge,tail_width=1.",
                                fc=(1.0, 0.7, 0.7), ec="none",
                                patchA=None,
                                patchB=el,
                                relpos=(0.2, 0.5),
                                )
                )
    """
    # flows
    black = (0,0,0,1)
    green = (1,1,0,1)
    collection = PatchCollection(flow_patches, 
                                 linewidths = (1,),
                                 facecolors = (green,),
                                 edgecolors = (black,),
                                 cmap=matplotlib.cm.jet, 
                                 alpha=0.8)
    ax.add_collection(collection)
    """
    # title
    plt.text(5917.9,1352690,"Travel Flow Map from Geo-tagged Photos (California)",family="san-serif",size=18)
    # legend
    legend_patches = []
    box = matplotlib.patches.Rectangle([786652,651074],1075860-786652,1261290-651074,ec=black,facecolor=None)
    plt.text(790000,1226340,"Legend",family="san-serif",size=12)
    parcel = matplotlib.patches.Rectangle([790000,1200000],20000,10000,ec=black,fc='gray')
    legend_patches.append(box)
    legend_patches.append(parcel)
    collection = PatchCollection(legend_patches, cmap=matplotlib.cm.jet )
    """
    plt.savefig('Result%s\\flow_%s_%s.png'%(roi_filename[18:20],
                                            roi_filename[18:20],
                                            roi_filename[21:21+roi_filename[21:].index('.')]),
                dpi=300,format='png')
    plt.show()

def save_flow(flows,flows_3,flows_4):
    o = open('Result%s\\flow_%s_%s.csv'%(roi_filename[18:20],
                                        roi_filename[18:20],
                                        roi_filename[21:21+roi_filename[21:].index('.')]),
             'w')
    for flow in flows:
        o.write('%s-%s,%s\n' % (flow[0],flow[1],flows[flow]))
    for flow in flows_3:
        o.write('%s-%s-%s,%s\n' % (flow[0],flow[1],flow[2],flows_3[flow]))
    for flow in flows_4:
        o.write('%s-%s-%s-%s,%s\n' % (flow[0],flow[1],flow[2],flow[3],flows_4[flow]))
    o.close()

def dump_flow(flows):
    output = open(roi_filename[:-9]+'flow.pkl', 'wb')
    pickle.dump(flows, output)
    output.close()
    
if __name__ == "__main__":
    roi_filename = sys.argv[1]
    print roi_filename
    min_sup = 0.01
    travel_routes = regenerate_travel_route('SourceData\\panoramio.csv',roi_filename,day_constrain=7)
    flows,flows_3,flows_4 = mining_travel(travel_routes, min_sup=min_sup)
    save_flow(flows,flows_3,flows_4)
    dump_flow(flows)
    plot_travel_pattern(roi_filename,flows)
