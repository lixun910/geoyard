import pickle, sys
import numpy as np
from datetime import datetime, date, time
from shapely.geometry import *
from ShapeHelper import *

def _get_roi_from_shp(filename):
    shape_type, shapeobjects = read_shapeobjects(filename)
    return shapeobjects

def regenerate_travel_route(photo_filename,roi_filename,day_constrain=7,save=False):
    roi_set = _get_roi_from_shp(roi_filename)
    travel_routes = {}
    roi_photo_count =np.zeros(len(roi_set)) 
    
    ############################
    # points file
    f = open(photo_filename)
    line = f.readline()
    line = f.readline()
    while(len(line)>0):
        line = line.strip().split(',')
        user = line[1]
        lat = float(line[2])
        lon = float(line[3])
        dt = line[4][2:]
        ptime = datetime.strptime(dt,"%y/%m/%d %H:%M:%S")
       
        # California
        if lon > -124.74 and lon < -113.99 and lat >32.417 and lat < 42.04:
        # LA
        #if lon > -119.18 and lon < -116.21 and lat > 32.538 and lat < 34.825:
        # SF
        #if lon > -123.4 and lon < -120.45 and lat > 36.395 and lat < 39.028:
            if not travel_routes.has_key(user):
                travel_routes[user] = []        
            travel_routes[user].append([lon,lat,ptime])
        line = f.readline()
      
    ##############################
    # regenerate travel route
    for userid in travel_routes.keys():
        # sorted in chrononical order
        route = sorted(travel_routes[userid], key=lambda route:route[2])
        # time constrain
        new_routes = []
        num_photos = len(route)
        count = 0
        while count < num_photos:
            photo = route[count]
            first_photo = photo
            tc_route = []
            tc_route.append(first_photo)
            count += 1
            if count >= num_photos:
                continue
            next_photo = route[count]

            while (next_photo[2] - first_photo[2]).days < day_constrain and count < num_photos-1:
                if photo[0] != next_photo[0] or photo[1] != next_photo[1]:
                    tc_route.append(next_photo)
                photo = next_photo
                count += 1
                next_photo = route[count]
                
            new_tc_route = []
            #convert to roi
            for p in tc_route:
                point = Point(p[0],p[1])
                for i in range(len(roi_set)):
                    if roi_set[i].contains(point):
                        roi_photo_count[i] += 1
                        if len(new_tc_route) == 0 or i != new_tc_route[-1]:
                            new_tc_route.append(i)
                        break
               
            if len(new_tc_route) > 0:
                new_routes.append(new_tc_route)
            
        travel_routes[userid] = new_routes
        #print userid, new_routes

    if save == True:
        output = open('Output/travel_routes.pkl', 'wb')
        pickle.dump(travel_routes, output)
        output.close()
        
    return roi_set,roi_photo_count, travel_routes

def generate_pajek_data(roi_set, roi_photo_count, travel_routes):
    """
    travel_routes: id,[[route],[route],...]
    route: roi->roi->roi
    
    point size:4-20
    line size :1-10
    arrow size:7-17
    """
    o = open('Output/travel19.NET','w')
    
    # Vertices info
    num_vertices = len(roi_set)
    o.write('*Vertices%4d\n' % num_vertices)
   
    xs,ys = [],[]
    for i,roi in enumerate(roi_set):
        xs.append(roi.centroid.x)
        ys.append(roi.centroid.y)
    x_range = abs(min(xs)-max(xs))
    y_range = abs(min(ys)-max(ys))
   
    vert_size = []
    vert_num_min, vert_num_max = min(roi_photo_count),max(roi_photo_count)
    vert_num_range = vert_num_max - vert_num_min
    _xs,_ys = [],[]
    for i in range(num_vertices):
        _xs.append( (xs[i] - min(xs) ) / x_range)
        _ys.append( 1-(ys[i] - min(ys) ) / y_range)
        vert_size.append( 8 + 16*( roi_photo_count[i] -  vert_num_min )/vert_num_range)
        
    for i,roi in enumerate(roi_set):
        o.write('%7d \"roi_%d\"    %.4f    %.4f    %.4f    s_size %d\n' % 
                (i+1,i+1,_xs[i],_ys[i],0.0,vert_size[i])) 
        
    # Arcs Infor
    o.write('*Arcs\n')
    
    dist_matrix = np.zeros([num_vertices,num_vertices])
    for id,routes in travel_routes.iteritems():
        for route in routes:
            for i in range(len(route)-1):
                start,end = route[i],route[i+1]
                dist_matrix[start][end] += 1
     
    arc_weight_min = 1000
    arc_weight_max = 0
    for i in range(num_vertices):
        for j in range(num_vertices):
            item = dist_matrix[i][j]
            if 0<item < arc_weight_min: 
                arc_weight_min = item
            if item > arc_weight_max:
                arc_weight_max = item
                
    arc_weight_range = arc_weight_max-arc_weight_min
    
    for i in range(num_vertices):
        for j in range(num_vertices):
            if dist_matrix[i][j] > 0:
                line_size = 1 + 9*( dist_matrix[i][j] - arc_weight_min ) / arc_weight_range 
                arrow_size = 1+ 8*( dist_matrix[i][j] - arc_weight_min ) / arc_weight_range 
                o.write('%7d%7d%8d w %d s %d\n' % 
                        (i+1,j+1,dist_matrix[i][j], line_size,arrow_size))
            
    o.write('*Edges\n')
    edge_weight_min = 1000
    edge_weight_max = 0
    for i in range(num_vertices):
        for j in range(i,num_vertices):
            item = dist_matrix[i][j] + dist_matrix[j][i]
            dist_matrix[i][j] = item
            if 0<item < edge_weight_min: 
                edge_weight_min = item
            if item > edge_weight_max:
                edge_weight_max = item
                
    edge_weight_range = edge_weight_max-edge_weight_min
    
    for i in range(num_vertices):
        for j in range(i,num_vertices):
            if dist_matrix[i][j] > 0:
                line_size = 1 + 9*( dist_matrix[i][j] - edge_weight_min ) / edge_weight_range 
                arrow_size = 1+ 8*( dist_matrix[i][j] - edge_weight_min ) / edge_weight_range 
                o.write('%7d%7d%8d w %d s %d\n' % 
                        (i+1,j+1,dist_matrix[i][j], line_size,arrow_size))
    o.close()
    
if __name__ == "__main__":
    roi_set, roi_photo_count, travel_routes = regenerate_travel_route('SourceData/panoramio.csv', 
                                           'Cluster25/Cluster_25.19.alpha.shp',
                                            day_constrain=7,
                                            save = True)
    generate_pajek_data(roi_set,roi_photo_count, travel_routes)
    