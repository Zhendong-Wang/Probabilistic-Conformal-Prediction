import folium
import matplotlib.pyplot as plt
import geopy.distance
from folium import plugins
from folium.plugins import HeatMap
import numpy as np
import geojson
from shapely.geometry import Point, Polygon

import geopandas as gpd


def dumb_hm_counting(points_origin, qt, grid_size = 100, caltype = 'uniform', density = None):
    points_origin = np.array(points_origin)
    # get a bounding box
    if caltype == 'uniform' or caltype == 'filtered':
        qtmax = qt
    elif caltype == 'density':
        # loop to find max round
        density = np.array(density)
        qtmax = qt * max(density)
    x_min = np.min(points_origin[:,0] - qtmax)
    x_max = np.max(points_origin[:,0] + qtmax)
    y_min = np.min(points_origin[:,1] - qtmax)
    y_max = np.max(points_origin[:,1] + qtmax)
    area = (x_max - x_min) * (y_max - y_min)
    # get grid inside
    xgrid = np.linspace(x_min, x_max, grid_size)
    ygrid = np.linspace(y_min, y_max, grid_size)
    xg, yg = np.meshgrid(xgrid, ygrid)
    total_count = 0
    area_count = 0
    lat = []
    lng = []
    for xc, yc in zip(xg.reshape(-1), yg.reshape(-1)):
        # whether this
        total_count += 1
        for ind, i in enumerate(points_origin):
            if caltype == 'uniform' or caltype == 'filtered':
                qt_i = qt
            elif caltype == 'density':
                qt_i = qt * density[ind]
            if sum((i-np.array([xc,yc]))**2) < qt_i:
            	lat.append(yc)
            	lng.append(xc)
    return lat, lng

def visual_pcp(pred, qt, Y_test, X_test = None, num = 10, seed = 0, caltype = 'uniform', density = None):
	# estimate the distance of qt
	a = np.array(pred[0][0])
	a[0] -= np.sqrt(qt)
	dist = geopy.distance.distance(np.array(pred[0][0]), a).meters

	for index in range(num):
		map_1 = folium.Map(location=[Y_test[index][1], Y_test[index][0]],tiles='OpenStreetMap',
		 zoom_start=12)
		map_hm = folium.Map(location=[Y_test[index][1], Y_test[index][0]],tiles='OpenStreetMap',
		 zoom_start=12)
		if X_test is not None:
			each = index
			folium.Marker([X_test[each, -1], X_test[each, -2]], popup=str(each)+': '+'pickup', icon=folium.Icon(color='red')).add_to(map_1)
			folium.Marker([Y_test[each, 1], Y_test[each, 0]], popup=str(each)+': '+'dropoff', icon=folium.Icon(color='blue')).add_to(map_1)
			folium.PolyLine([[X_test[each, -1], X_test[each, -2]],[Y_test[each, 1], Y_test[each, 0]]], color="red", weight=2.5, opacity=1).add_to(map_1)			

		for ind, i in enumerate(pred[index]):
			if caltype == 'uniform' or caltype == 'filtered':
				folium.Circle(location=[i[1], i[0]], popup='Point 1A', radius=dist, fill=True, fill_color='#ffff00', fill_opacity=0.2).add_to(map_1)
				#lat, lng = dumb_hm_counting(pred[index], qt, grid_size = 100, caltype = caltype)
			elif caltype == 'density':
				folium.Circle(location=[i[1], i[0]], popup='Point 1A', radius=dist*np.sqrt(density[index][ind]), fill_color='#ffff00', fill_opacity=0.2).add_to(map_1)
				#lat, lng = dumb_hm_counting(pred[index], qt, grid_size = 100, caltype = caltype, density = density[index])
		#HeatMap(list(zip(lat, lng)), min_opacity=0.05).add_to(map_hm)
		map_1.save(f'./fig/taxi/{seed}_pcp_{caltype}_map{index}.html')
		#map_hm.save(f'./fig/taxi/{seed}_hmpcp_{caltype}_map{index}.html')

def visual_subgroup(gj, neighbor_list, pred, qt, Y_test, num = 10, seed = 0, caltype = 'uniform', density = None):
	a = np.array(pred[0][0])
	a[0] -= np.sqrt(qt)
	dist = geopy.distance.distance(np.array(pred[0][0]), a).meters
	for k in range(gj.shape[0]):
		name = gj.loc[k,'name']
		print(f'plot {name}')
		map_1 = folium.Map(location=[40.772747, -73.978386],tiles='OpenStreetMap',
		 zoom_start=12)
		map_hm = folium.Map(location=[40.772747, -73.978386],tiles='OpenStreetMap',
		 zoom_start=12)
		index_thisregion = np.where(np.array(neighbor_list) == k)[0]
		latsub = []
		lngsub = []
		for j in index_thisregion:
			for ind, i in enumerate(pred[j]):
				if caltype == 'uniform' or caltype == 'filtered':
					folium.Circle(location=[i[1], i[0]], popup='Point 1A', radius=dist, fill=True, fill_color='#ffff00', fill_opacity=0.2).add_to(map_1)
				elif caltype == 'density':
					folium.Circle(location=[i[1], i[0]], popup='Point 1A', radius=dist*np.sqrt(density[index][ind]), fill_color='#ffff00', fill_opacity=0.2).add_to(map_1)
			lat, lng = dumb_hm_counting(pred[j], qt, grid_size = 100, caltype = caltype, density = density)
			latsub += lat
			lngsub += lng
		HeatMap(list(zip(latsub, lngsub)), min_opacity=0.05).add_to(map_hm)
		map_1.save(f'./fig/taxi/{seed}_pcp_{caltype}_map_{name}.html')
		map_hm.save(f'./fig/taxi/{seed}_hmpcp_{caltype}_map_{name}.html')


def visual_subgroup_cd(gj, neighbor_list, pred_y0, pred_y1, Y_test, num = 10, seed = 0):
	for k in range(gj.shape[0]):
		name = gj.loc[k,'name']
		map_1 = folium.Map(location=[40.772747, -73.978386],tiles='OpenStreetMap',
		 zoom_start=12)
		index_thisregion = np.where(np.array(neighbor_list) == k)[0]
		latsub = []
		lngsub = []
		for j in index_thisregion:
			for x_ in pred_y1[j]:
				for y_ in pred_y0[j]:
					folium.Rectangle(bounds=np.array([[x_[0],y_[0]],[x_[1],y_[1]]]), \
						color='#ff7800', fill=True, fill_color='#ffff00', fill_opacity=0.2).add_to(map_1)
		map_1.save(f'./fig/taxi/{seed}_cd_map_{name}.html')


def visual_subgroup_baselines(gj, neighbor_list, pred_y0, pred_y1, Y_test, num = 10, method_name = 'chr', seed = 0):
	for k in range(gj.shape[0]):
		name = gj.loc[k,'name']
		map_1 = folium.Map(location=[40.772747, -73.978386],tiles='OpenStreetMap',
		 zoom_start=12)
		index_thisregion = np.where(np.array(neighbor_list) == k)[0]
		latsub = []
		lngsub = []
		for j in index_thisregion:
			folium.Rectangle(bounds=np.array([[pred_y1[j][0],pred_y0[j][0]],[pred_y1[j][1],pred_y0[j][1]]]), \
					color='#ff7800', fill=True, fill_color='#ffff00', fill_opacity=0.2).add_to(map_1)
		map_1.save(f'./fig/taxi/{seed}_{method_name}_map_{name}.html')



def visual_baseline(pred_y0, pred_y1, Y_test, X_test = None, num = 10, method_name = 'chr', seed = 0):
	for index in range(num):
		map_1 = folium.Map(location=[Y_test[index][1], Y_test[index][0]],tiles='OpenStreetMap',
		 zoom_start=12)
		each = index 
		if X_test is not None:
			folium.Marker([X_test[each, -1], X_test[each, -2]], popup=str(each)+': '+'pickup', icon=folium.Icon(color='red')).add_to(map_1)
			folium.Marker([Y_test[each, 1], Y_test[each, 0]], popup=str(each)+': '+'dropoff', icon=folium.Icon(color='blue')).add_to(map_1)
			folium.PolyLine([[X_test[each, -1], X_test[each, -2]],[Y_test[each, 1], Y_test[each, 0]]], color="red", weight=2.5, opacity=1).add_to(map_1)			
		folium.Rectangle(bounds=np.array([[pred_y1[index][0],pred_y0[index][0]],[pred_y1[index][1],pred_y0[index][1]]]), color='#ff7800', fill=True, fill_color='#ffff00', fill_opacity=0.2).add_to(map_1)
		map_1.save(f'./fig/taxi/{seed}_{method_name}_map{index}.html')


def visual_baseline_cd(pred_y0, pred_y1, Y_test, X_test = None, num = 10, seed = 0):
	for index in range(num):
		map_1 = folium.Map(location=[Y_test[index][1], Y_test[index][0]],tiles='OpenStreetMap',
		 zoom_start=12)
		if X_test is not None:
			each = index 
			folium.Marker([X_test[each, -1], X_test[each, -2]], popup=str(each)+': '+'pickup', icon=folium.Icon(color='red')).add_to(map_1)
			folium.Marker([Y_test[each, 1], Y_test[each, 0]], popup=str(each)+': '+'dropoff', icon=folium.Icon(color='blue')).add_to(map_1)
			folium.PolyLine([[X_test[each, -1], X_test[each, -2]],[Y_test[each, 1], Y_test[each, 0]]], color="red", weight=2.5, opacity=1).add_to(map_1)			
		for x_ in pred_y1[index]:
			for y_ in pred_y0[index]:
				folium.Rectangle(bounds=np.array([[x_[0],y_[0]],[x_[1],y_[1]]]), \
					color='#ff7800', fill=True, fill_color='#ffff00', fill_opacity=0.2).add_to(map_1)
		map_1.save(f'./fig/taxi/{seed}_cd_map{index}.html')


def visual_data(X_test, Y_test, num = 20, seed = 0):
	loc = X_test[0,-2],X_test[0,-1]
	map_1 = folium.Map(location=[Y_test[0][1], Y_test[0][0]],tiles='OpenStreetMap',
		 zoom_start=12)
	for each in range(num):
		if each in [115, 71, 25, 0, 118]:
			folium.Marker([X_test[each, -1], X_test[each, -2]], popup=str(each)+': '+'pickup', icon=folium.Icon(color='red')).add_to(map_1)
			folium.Marker([Y_test[each, 1], Y_test[each, 0]], popup=str(each)+': '+'dropoff', icon=folium.Icon(color='blue')).add_to(map_1)
			poly = folium.PolyLine([[X_test[each, -1], X_test[each, -2]],[Y_test[each, 1], Y_test[each, 0]]], color="red", weight=2.5, opacity=1, stroke = True)
			poly.add_to(map_1)
			arrows = getArrows(locations=[[X_test[each, -1], X_test[each, -2]],[Y_test[each, 1], Y_test[each, 0]]], color = 'red', n_arrows=1)
			for arrow in arrows:
				arrow.add_to(map_1)
	map_1.save(f'./fig/taxi/{seed}_data_map.html')

from collections import namedtuple
def getArrows(locations, color='red', size=6, n_arrows=3):
    
    '''
    Get a list of placed and rotated arrows or markers to be plotted
    
    Parameters
    locations : list of lists of latitude longitude that represent the begining and end of Line. 
                    this function Return list of arrows or the markers
    '''
    
    Point = namedtuple('Point', field_names=['lat', 'lon'])
    
    # creating point from Point named tuple
    point1 = Point(locations[0][0], locations[0][1])
    point2 = Point(locations[1][0], locations[1][1])
    
    # calculate the rotation required for the marker.  
    #Reducing 90 to account for the orientation of marker
    # Get the degree of rotation
    angle = get_angle(point1, point2) - 90
    
    # get the evenly space list of latitudes and longitudes for the required arrows

    arrow_latitude = np.linspace(point1.lat, point2.lat, n_arrows + 2)[1:n_arrows+1]
    arrow_longitude = np.linspace(point1.lon, point2.lon, n_arrows + 2)[1:n_arrows+1]
    
    final_arrows = []

    color = 'red'
    
    #creating each "arrow" and appending them to our arrows list
    for points in zip(arrow_latitude, arrow_longitude):
        final_arrows.append(folium.RegularPolygonMarker(location=points, 
                      color = color, fill_color=color, number_of_sides=3, 
                      radius=size, rotation=angle))
    return final_arrows

def get_angle(p1, p2):
    
    '''
    This function Returns angle value in degree from the location p1 to location p2
    
    Parameters it accepts : 
    p1 : namedtuple with lat lon
    p2 : namedtuple with lat lon
    
    This function Return the vlaue of degree in the data type float
    
    Pleae also refers to for better understanding : https://gist.github.com/jeromer/2005586
    '''
    
    longitude_diff = np.radians(p2.lon - p1.lon)
    
    latitude1 = np.radians(p1.lat)
    latitude2 = np.radians(p2.lat)
    
    x_vector = np.sin(longitude_diff) * np.cos(latitude2)
    y_vector = (np.cos(latitude1) * np.sin(latitude2) 
        - (np.sin(latitude1) * np.cos(latitude2) 
        * np.cos(longitude_diff)))
    angle = np.degrees(np.arctan2(x_vector, y_vector))
    
    # Checking and adjustring angle value on the scale of 360
    if angle < 0:
        return angle + 360
    return angle    
