import pickle, sys
from pylab import rand
from datetime import datetime, date, time
from shapely.geometry import *
from ShapeHelper import *

def _get_roi_from_shp(filename):
    shape_type, shapeobjects = read_shapeobjects(filename)
    return shapeobjects

def get_roi_hits(photo_filename, roi_filename):
    roi_set = _get_roi_from_shp(roi_filename)
    roi_users = [[] for i in range(len(roi_set))]
    roi_hits = range(len(roi_set))
    
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
       
        if lon > -124.74 and lon < -113.99 and lat >32.417 and lat < 42.04:
            point = Point(lon,lat)
            for i in range(len(roi_set)):
                if roi_set[i].contains(point):
                    roi_users[i].append(user)
                    break
        line = f.readline()
    f.close()
    
    for i in range(len(roi_set)):
        roi_hits[i] = len(set(roi_users[i]))
        
    return roi_hits

def photo_dist(p1,p2):
    import math
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] -p2[1])**2)*100000

def get_travel_interval(photo_filename):
    travel_routes = {}

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
       
        if lon > -124.74 and lon < -113.99 and lat >32.417 and lat < 42.04:
            if not travel_routes.has_key(user):
                travel_routes[user] = []        
            travel_routes[user].append([lon,lat,ptime])
        line = f.readline()
    f.close()  
    
    ##############################
    # regenerate travel route
    
    o = open('travel_distance.txt','w')
    solo_trip_intervals_day = []
    solo_trip_intervals_hours = []
    travel_distance_day_1 = []
    travel_distance_day_2 = []
    travel_distance_day_3 = []
    
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
            tc_route = [] # solo trip
            tc_route.append(first_photo)
            count += 1
            if count >= num_photos:
                continue
            next_photo = route[count]

            while (next_photo[2] - first_photo[2]).days <= 2 and count < num_photos-1:
                if photo[0] != next_photo[0] or photo[1] != next_photo[1]:
                    tc_route.append(next_photo)
                photo = next_photo
                count += 1
                next_photo = route[count]
                
            if len(tc_route) > 1:
                new_routes.append(tc_route)
                interval = tc_route[-1][2] - tc_route[0][2]
                solo_trip_intervals_day.append(interval.days+1)
                solo_trip_intervals_hours.append(interval.seconds/60.0/60.0)
                
                # 1-day-trip travel distance
                if interval.days == 0:
                    num_places = len(tc_route)
                    for i in range(num_places-1):
                        d = photo_dist(tc_route[i], tc_route[i+1])
                        travel_distance_day_1.append(d)
                elif interval.days == 1:
                    num_places = len(tc_route)
                    for i in range(num_places-1):
                        d = photo_dist(tc_route[i], tc_route[i+1])
                        travel_distance_day_2.append(d)
                elif interval.days == 2:
                    num_places = len(tc_route)
                    for i in range(num_places-1):
                        d = photo_dist(tc_route[i], tc_route[i+1])
                        travel_distance_day_3.append(d)

                
        travel_routes[userid] = new_routes
    print solo_trip_intervals_day
    print travel_distance_day_1
    print travel_distance_day_2
    print travel_distance_day_3
    o.write('%s\n' % solo_trip_intervals_day)
    o.write('%s\n' % travel_distance_day_1)
    o.write('%s\n' % travel_distance_day_2)
    o.write('%s\n' % travel_distance_day_3)
    o.close()
    return travel_routes
    
def regenerate_travel_route(photo_filename,roi_filename,day_constrain=7,save=False):
    roi_set = _get_roi_from_shp(roi_filename)
    travel_routes = {}
    
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
                        if len(new_tc_route) == 0 or i != new_tc_route[-1]:
                            new_tc_route.append(i)
                        break
            """
            #assig to nearest roi
            roi = -1
            prev_max_dist = 1000000000
            for p in tc_route:
                point = Point(p[0],p[1])
                for i in range(len(roi_set)):
                    cur_dist = point.distance(roi_set[i]) 
                    if cur_dist == 0:
                        roi = i 
                        break
                    elif cur_dist < prev_max_dist:
                        roi = i
                        prev_max_dist = cur_dist
                if len(new_tc_route) == 0 or i != new_tc_route[-1]:
                    new_tc_route.append(i)
            """         
            if len(new_tc_route) > 0:
                new_routes.append(new_tc_route)
            
        travel_routes[userid] = new_routes
        #print userid, new_routes

    if save == True:
        output = open('travel_routes.pkl', 'wb')
        pickle.dump(travel_routes, output)
        output.close()
        
    return travel_routes

def find_subsequences(travel_routes,k,min_sup=0.0051):
    k_candidates = {}
    for user in travel_routes:
        seqs = travel_routes[user]
        candidates = {}
        for seq in seqs:    
            for i in range(len(seq)-k+1):
                cand = tuple(seq[i:i+k])
                candidates[cand] = True
                """
                if k_candidates.has_key(cand):
                    k_candidates[cand] += 1
                else:
                    k_candidates[cand] = 1
                """
        
        # calc support for each candidate
        for cand in candidates:
            if k_candidates.has_key(cand):
                k_candidates[cand] += 1
            else:
                k_candidates[cand] = 1
        
    # support value
    m = len(travel_routes)
    candidates = k_candidates.keys()
    for cand in candidates:
        support = float(k_candidates[cand]) / m
        #print cand,'/',m,'=',support
        k_candidates[cand] = support
            
    return k_candidates
    
def apriori_gen(k_sequences):
    k_plus_1_candidates = {}
    for user in travel_routes:
        seqs = travel_routes[user]
        candidates = {}
        for seq in seqs:    
            for i in range(len(seq)-k+1):
                cand = tuple(seq[i:i+k])
                candidates[cand] = True
    
    for seq in k_sequences:
        pass
    
def mining_travel(travel_routes, min_sup=0.01, load=False,load_filename=''):
    if load == True:
        pkl_file = open(load_filename,'rb')
        travel_routes = pickle.load(pkl_file)
        pkl_file.close()
    
    # remove empty user
    users = travel_routes.keys()
    for u in users:
        if len(travel_routes[u]) <= 1:
            travel_routes.pop(u)
    print "user count:", len(travel_routes)
    
    # pattern discovery
    F_k = find_subsequences(travel_routes,2)    
    F_k_1 = find_subsequences(travel_routes,3)
    F_k_2 = find_subsequences(travel_routes,4)
    """
    while len(F_k) > 0:
        k = k + 1
        # generate candidate k-subsequences
        C_k = apriori_gen(F_k)
        for t in travel_routes:
            # identify all candidates contained in t.
            C_t = subsequence(C_k,t)
            for c in C_t:
                # increment the support count
                pass
        # extract the frequent k-subsequences
        F_k = find_subsequences(,k)
    """
    return F_k
           

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
    
powerlaw = lambda x, alpha, minx: (alpha-1) / minx * (x/minx)**-alpha

def plot_pl(dataset,alpha,xmin):        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.plot(range(1,len(ranks)+1),ranks,"o",alpha=0.9, markersize=4, mfc=rand(3))
    ax.plot(data, powerlaw(data, alpha, xmin),"o")
    ax.set_title('powerlaw plot')
    ax.set_xlabel('x (# visitors)')
    ax.set_ylabel('p(x)')
    
    for data in dataset:
        x = np.sort(data)
        n = len(x)
        xcdf = np.arange(n, 0, -1, dtype='float') / float(n)
        q = x[x>=xmin]
        fcdf = (q/xmin)**(1-alpha)
        nc = xcdf[np.argmax(x>=xmin)]
        
        plotx = np.linspace(q.min(),q.max(),1000)
        ploty = (plotx/xmin)**(1-alpha) * nc
        
        color = rand(3)
        ax.loglog(x,xcdf, 'o', alpha='0.6',mew='3',mfc='w',ms=8,mec=color)
        ax.loglog(plotx,ploty,ls='--',c=color)
    plt.show()

if __name__ == "__main__":
    if 1==1:
        get_travel_interval('SourceData\\panoramio.csv')
    if 1==0:
        o = open('output.txt','w')
        for i in range(1,30):
            hits = get_roi_hits('SourceData\\panoramio.csv', 'Cluster25\\Cluster_25.%d.alpha.shp' % i)
            print i,
            print hits
            o.write('%d,%s\n' %(i,str(hits)))
        o.close()
    
    import plfit
    hits = [42, 35, 24, 25, 41, 20, 30, 22, 37, 45, 53, 52, 14, 52, 51, 32, 14, 30, 31, 88, 8, 54, 67, 48, 14, 80, 32, 35, 31, 57, 34, 22, 15, 74, 8, 44, 32, 22, 339, 59, 117, 63, 127, 47, 200, 39]
    hits = [42, 35, 24, 25, 41, 20, 30, 22, 37, 45, 53, 52, 14, 52, 51, 32, 14, 30, 31, 88, 8, 54, 67, 48, 14, 80, 32, 35, 31, 
57, 34, 22, 15, 74, 8, 44, 32, 22, 59, 63, 47, 39, 64, 47, 80, 54, 12, 8, 27, 45, 47, 45, 38, 48, 60, 61, 75, 62, 40, 
68, 61, 19, 10, 16, 111, 21, 13, 32, 31, 92, 88]
    hits.sort(reverse=True)
    pl = plfit.plfit(hits, quiet=False, silent=True)
    pl.plotcdf()
    
    ######################################
    min_sup = 0.01
    ranks = []
    for i in range(11,12):
        travel_routes = regenerate_travel_route('SourceData\\panoramio.csv', 
                                                'Cluster25\\Cluster_25.%d.alpha.shp'% i,
                                                day_constrain=7,
                                                save = True)
        F_K = mining_travel(travel_routes, min_sup=min_sup,load=False, load_filename='travel_routes.pkl')
        ranks.append(F_K.values())
    myplfit = plfit.plfit(ranks[0],quiet=True,silent=True)
    myplfit.plotcdf()
    plfit.pylab.show()
    plot_rank_size(ranks)