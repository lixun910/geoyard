from mpl_toolkits.basemap import Basemap
import pylab
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Wedge, Polygon,Rectangle
from matplotlib.collections import PatchCollection
import numpy as np
import string,copy,datetime,math
import shapely.geometry as shapely
from ShapeHelper import *
from TravelMining import *

import aspen
from aspen.handlers.static import wsgi as static_handler
from aspen.utils import translate



def load_flow_in_memory(flow_filename):
    pkl_file = open(flow_filename,'rb')
    flows = pickle.load(pkl_file)
    pkl_file.close()
    return flows

def load_roi_in_memory(roi_filename):
    shape_type, cluster_polygons = read_objects(roi_filename)
    return cluster_polygons

# default loadup
m = Basemap(llcrnrlon=-124.74,llcrnrlat=32.417,urcrnrlon=-113.99,urcrnrlat=42.04,resolution='c',projection='merc')    
global_flow_map = {}
global_roi_map = {}
for i in [1,10,19]:
    flow_file_name = './Cluster25/Cluster_25.%s.flow.pkl' % i
    global_flow_map[i] = load_flow_in_memory(flow_file_name)
    roi_file_name = './Cluster25/Cluster_25.%s.alpha.shp' % i
    global_roi_map[i] = load_roi_in_memory(roi_file_name)

    
def project_polygon(polygons):
    tmp_polygons = []
    for i,polygon in enumerate(polygons):
	tmp_polygon = []
        for j,p in enumerate(polygon):
            tmp_polygon.append(list(m(p[0],p[1])))
	tmp_polygons.append(tmp_polygon)
    return tmp_polygons

def gen_centroid_shape(center_point,radius):
    centroid_shape = []
    # default points 36, start from (0,0)
    theta = 2*math.pi / 36.0
    for i in range(36):
	x = radius * math.cos(theta*i) + center_point[0]
	y = radius * math.sin(theta*i) + center_point[1]
	centroid_shape.append(list(m(x,y,inverse=True)))
    return centroid_shape

def gen_travel_flow(_start, _end, weight, m,offset_xy,offset_z,arrow_weight):
    start = np.array(copy.deepcopy(_start))
    end = np.array(copy.deepcopy(_end))
    flow_arrow = []
    
    length = np.linalg.norm(end - start)
    cos_theta = (end - start)[0] / length
    sin_theta = (end - start)[1] / length
    
    offset_x = offset_xy * cos_theta
    offset_y = offset_xy * sin_theta
    
    start += [offset_x, offset_y]
    end -= [offset_x,offset_y]
    
    offset = offset_z
    thick = weight * arrow_weight
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
        x = p[0]*cos_theta - p[1]*sin_theta + start[0]
        y = p[0]*sin_theta + p[1]*cos_theta + start[1]
        flow_arrow[i] = list(m(x,y,inverse=True))
    
    return flow_arrow

def show_flows(environ, start_response):
    ROOT = aspen.paths.root
    path = environ['PATH_INFO']
    print path
    fspath = translate(ROOT, path)
    zoomLevel,bound = path[1:].split('/')
    zoomLevel = int(float(zoomLevel))
    
    centroid_radius = 0
    offset_xy = 0
    offset_z = 0
    arrow_weight = 0
    flow_filename = ''
    roi_filename = ''

    flows = None
    centroid = []
    centroid_patches = []
    flow_patches = []
    cluster_polygons = []
    
    print zoomLevel
    if 5<=zoomLevel<7:
	centroid_radius = 15000
	offset_xy = 25000
	offset_z = 2000
	arrow_weight = 100000
	flows = global_flow_map[1]
	cluster_polygons = global_roi_map[1]
	print 1
    elif 7<=zoomLevel<9:
	centroid_radius = 1500
	offset_xy = 2500
	offset_z = 200
	arrow_weight =30000
	flows = global_flow_map[10]
	cluster_polygons = global_roi_map[10]
	print 2
    elif zoomLevel >=9:
	centroid_radius = 150
	offset_xy = 250
	offset_z = 20
	arrow_weight = 3000
	flows = global_flow_map[19]
	cluster_polygons = global_roi_map[19]
	print zoomLevel > 14
    dt = datetime.now()
    print dt.strftime("%d/%m/%y %H:%M:%S")
    
    polygons = project_polygon(cluster_polygons)
    for polygon in polygons:
	center = shapely.Polygon(polygon).centroid
	centroid.append([center.x,center.y])	
    for flow in flows:
	flow_patches.append(gen_travel_flow(centroid[flow[0]],centroid[flow[1]],flows[flow],m,offset_xy,offset_z,arrow_weight))
    
    dt = datetime.now()
    print dt.strftime("%d/%m/%y %H:%M:%S")
    for i,p in enumerate(centroid):
	centroid_patches.append(gen_centroid_shape(centroid[i],centroid_radius))
	centroid_patches.append(gen_centroid_shape(centroid[i],centroid_radius/3.0))
	centroid[i] = list(m(p[0],p[1],inverse=True))
		
    # write back to http client
    str_centroids = ';'.join('&'.join(','.join(str(i) for i in p) for p in polygon) for polygon in centroid_patches)
    str_clusters = ';'.join('&'.join(','.join(str(i) for i in p) for p in polygon) for polygon in cluster_polygons)
    str_flows = ';'.join('&'.join(','.join(str(i) for i in p) for p in polygon) for polygon in flow_patches)
    response = '%s|%s|%s' % (str_clusters,str_centroids,str_flows)
    
    dt = datetime.now()
    print dt.strftime("%d/%m/%y %H:%M:%S")
    
    start_response('200 OK', [('CONTENT-TYPE','text/plain')])    
    return response