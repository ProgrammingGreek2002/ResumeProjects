'''
Make sure to have csv in same directory

Install pip below
pip install pandas
'''
import pandas as pd
import math

#City object to store values from csv file
class City:
    def __init__(self, name, description, latitude, longitude):
        self.name = name
        self.description = description
        self.latitude = latitude
        self.longitude = longitude


#Function that calculates the euclidean distance between the cities
def calculate_distance(city1, city2):
    lat1, lon1 = city1.latitude, city1.longitude
    lat2, lon2 = city2.latitude, city2.longitude
    distance = math.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2)
    return distance


#Function that creates the distance matrix to be used in the depth search
def distance_matrix_dfs(cities):
    numberOfCities = len(cities)
    distance = [[0] * numberOfCities for _ in range(numberOfCities)]
    for i in range(numberOfCities):
        for j in range(i + 1, numberOfCities):
            distanceNum = calculate_distance(cities[i], cities[j])
            distance[i][j] = distanceNum
            distance[j][i] = distanceNum
    return distance


#Opening the csv data file and storing the values in the city object using the pandas library
city_data = pd.read_csv('city_data_50.csv')
cities = []

for _, row in city_data.iterrows():
    city = City(row['name'], row['description'], row['latitude'], row['longitude'])
    cities.append(city)


#Depth search for shortest path (iterative). Close the tour back to start.
def depth(start_city, distance_matrix):
    n = len(distance_matrix)
    stack = [(start_city, [start_city], 0.0)]
    s_path = None
    min_distance = float('inf')

    while stack:
        current_city, path, dist_so_far = stack.pop()

        if len(path) == n:
            # close the loop back to start
            total = dist_so_far + distance_matrix[current_city][start_city]
            if total < min_distance:
                min_distance = total
                s_path = path + [start_city]
            continue

        for next_city in range(n):
            if next_city not in path:
                next_path = path + [next_city]
                next_distance = dist_so_far + distance_matrix[current_city][next_city]
                stack.append((next_city, next_path, next_distance))

    return s_path, min_distance


#Very small, fast fallback for big N (keeps your structure intact)
def nearest_neighbor_tour(distance_matrix, start=0):
    n = len(distance_matrix)
    visited = [False] * n
    order = [start]
    visited[start] = True
    total = 0.0
    cur = start
    for _ in range(n - 1):
        #choose the closest unvisited
        nxt = min((j for j in range(n) if not visited[j]), key=lambda j: distance_matrix[cur][j])
        total += distance_matrix[cur][nxt]
        order.append(nxt)
        visited[nxt] = True
        cur = nxt
    #close tour
    total += distance_matrix[cur][start]
    order.append(start)
    return order, total


start = 0
dist_mat = distance_matrix_dfs(cities)

#Use exact DFS for small N; otherwise fallback so it completes on 50 cities
if len(cities) <= 12:
    sPath, best_dist = depth(start, dist_mat)
    search_name = "Depth-first Search"
else:
    sPath, best_dist = nearest_neighbor_tour(dist_mat, start)
    search_name = "Depth-first Search (nearest-neighbor fallback for N>12)"

print(search_name)
print("Shortest Path:")
for i in range(len(sPath) - 1):
    print(cities[sPath[i]].name, "->", end=" ")
print(cities[sPath[-1]].name)
print("Shortest Distance:", best_dist)



#bfs solver
def distance_matrix(location_one, location_two):
    #getting lat and long
    lat_one, long_one = location_one[:2]
    lat_two, long_two = location_two[:2]
    #distance calculation
    distance = math.sqrt((lat_two - lat_one) ** 2 + (long_two - long_one) ** 2)
    return distance


def create_matrix():
    list_size = len(location_list)
    #matrix creation
    matrix = [[0] * list_size for _ in range(list_size)]
    for i in range(list_size):
        for j in range(list_size):
            if i != j:
                matrix[i][j] = distance_matrix(location_list[i], location_list[j])
    return matrix


def bfs_algorithm(matrix, name_list):
    matrix_size = len(matrix)
    start = 0
    visited = [False] * matrix_size
    visited[start] = True
    visited_path = [(start, name_list[start])]
    starting_distance = 0.0

    def solve(curr_location, curr_path, curr_distance, shortest_distance, shortest_path):
        #base case
        if len(curr_path) == matrix_size:
            total_distance = curr_distance + matrix[curr_path[-1][0]][start]
            if total_distance < shortest_distance:
                shortest_distance = total_distance
                shortest_path = curr_path + [(start, name_list[start])]
            return shortest_distance, shortest_path

        for next_location in range(matrix_size):
            if not visited[next_location]:
                visited[next_location] = True
                shortest_distance, shortest_path = solve(
                    next_location,
                    curr_path + [(next_location, name_list[next_location])],
                    curr_distance + matrix[curr_location][next_location],
                    shortest_distance,
                    shortest_path
                )
                visited[next_location] = False
        return shortest_distance, shortest_path

    #initialize with +inf
    inf = float('inf')
    shortest_distance, shortest_path = solve(start, visited_path, starting_distance, inf, None)
    location_names = [location_name for _, location_name in shortest_path]
    return shortest_distance, location_names


#MAIN FUNCTION (second part)
df = pd.read_csv("city_data_50.csv")

#extract the values into list
location = df[['latitude', 'longitude']]
location_list = location.values.tolist()

#flatten to a plain list of strings
name_list = df['name'].tolist()

matrix = create_matrix()

#use exact only for small N; else use the same nearest-neighbor fallback
if len(name_list) <= 12:
    shortest_distance, shortest_path = bfs_algorithm(matrix, name_list)
    label = "Breadth-first search"
else:
    ids_path, nn_dist = nearest_neighbor_tour(matrix, start=0)
    shortest_distance = nn_dist
    shortest_path = [name_list[i] for i in ids_path]
    label = "Breadth-first search (nearest-neighbor fallback for N>12)"

print("\n" + label)
print("Shortest Distance:")
print(shortest_distance)
print("Shortest Path:")
print(shortest_path)
