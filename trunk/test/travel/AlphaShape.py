""" python 2.4"""
from CGAL.Alpha_shapes_2 import *
from CGAL.Triangulations_2 import Delaunay_triangulation_2
from CGAL.Kernel import *
from ShapeHelper import *
import numpy as np
import sys
    
def generate_alpha_shape(shapeobjects, cluster_id):
    points = []
    list_of_points = []
    
    num_of_objects = len(shapeobjects)
    for i in range(num_of_objects):
	if dbfobjects[i]['Cluster'] == cluster_id:
	    points.append(shapeobjects[i])
	    
    for i in range(len(points)):
	list_of_points.append(Point_2(points[i][0],points[i][1]))
    
    a = Alpha_shape_2()
    a.make_alpha_shape(list_of_points)
    a.set_mode(Alpha_shape_2.Mode.REGULARIZED)
    a.set_alpha(0.2)
    alpha_shape_edges = []
    alpha_shape_vertices = []

    pairs = {}
    for it in a.alpha_shape_edges:
	alpha_shape_edges.append(a.segment(it))
	pts = alpha_shape_edges[-1]
	pairs[pts[0].x(),pts[0].y()] = pts[1].x(),pts[1].y()
	
    for it in a.alpha_shape_vertices:
	alpha_shape_vertices.append(it)
	
    """
    print "alpha_shape_edges"	
    print len(alpha_shape_edges)
    print "alpha_shape_vertices"	
    print len(alpha_shape_vertices)
    print "Optimal alpha: " 
    print a.find_optimal_alpha(2).next()
    #show_alpha_values(a)
    """
    
    points = []
    start_point = pairs.keys()[0]
    points.append(start_point)
    count = 1
    while count <= len(alpha_shape_vertices):
	end_point = pairs[start_point]
	points.append(end_point)
	start_point = end_point
	count += 1
	
    # write to shape files
    hull = []
    for it in points:
	hull.append(it)
    return hull
    
    
def export_hull_shapefile(shapefilename,hull_set):
    shapeType = shapelib.SHPT_POLYGON
    shapeFile = shapelib.create(shapefilename, shapeType)
    dbfName = shapefilename[:-3]+ 'dbf'
    dbf = dbflib.create(dbfName)
    dbf.add_field('ID', dbflib.FTInteger, 50,0)
    dbf.add_field('Cluster', dbflib.FTInteger, 50,0)
    
    for i,hull in enumerate(hull_set): 
	cluster_id = i
	if cluster_id == len(hull_set): 
	    cluster_id = -1
	shapeObject = np.array(hull)
	n = len(shapeObject)
	shapeObject = np.append(shapeObject,[[.0] for j in range(n)],axis=1)
        shapeObject = [tuple(j) for j in shapeObject]
        obj = shapelib.SHPObject(shapeType, -1, [shapeObject])
        shapeFile.write_object(-1, obj)
        dbf.write_record(i, {'ID':i,'Cluster':cluster_id})
        
    shapeFile.close()
    dbf.close()
    
    print shapefilename
    
if __name__ == '__main__':
    base = sys.argv[1]
    for i in range(1,30):
	shapefilename  = '%s\\Cluster_%s.%s.shp' % (base[:-1], base[-3:-1],i)
	shape_type, shapeobjects, dbfobjects = read_objects(shapefilename, with_dbf_data = True)
	
	num_clusters = 0
	for i in dbfobjects:
	    num_clusters = max(num_clusters,i['Cluster'])
	    
	hull_set = []
	for i in range(num_clusters+1):
	    hull_set.append(generate_alpha_shape(shapeobjects,i))
	    
	export_hull_shapefile(shapefilename[:-3]+'alpha.shp',hull_set)

    