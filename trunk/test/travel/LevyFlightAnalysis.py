import pickle, sys
from datetime import datetime, date, time
from shapely.geometry import *
from ShapeHelper import *

def _get_roi_from_shp(filename):
    shape_type, shapeobjects = read_shapeobjects(filename)
    return shapeobjects

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
        if support < min_sup:
            k_candidates.pop(cand)
            pass
        else:
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
    return F_k,F_k_1,F_k_2
            
if __name__ == "__main__":
    min_sup = 0.01
    mining_travel(None, min_sup=min_sup,load=True, load_filename='travel_routes.pkl')
    
    travel_routes = regenerate_travel_route('SourceData\\panoramio.csv', 
                                            'Cluster25\\Cluster_25.1.alpha.shp',
                                            day_constrain=7,
                                            save = True)
    