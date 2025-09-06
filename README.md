**File Description**

**Traveling Salesman Solver**

Reads a CSV of cities (name, lat, long), builds a Euclidean distance matrix, and finds a shortest round-trip that visits 
every city. Uses an exact depth-first search for small datasets and automatically falls back to a nearest-neighbor heuristic 
when N > 12 to keep runtime reasonable. Includes a second solver variant (labeled “BFS”) and prints the tour order and 
total distance. (Educational demo; distances use simple Euclidean lat/long, not haversine.)

**K-Means Clustering from Scratch**

This mini project implements K-Means clustering (k=2) in pure Python to group city locations by latitude/longitude. 
It reads city_data_50.csv, uses Euclidean distance to iteratively assign points and update centroids until convergence,
and visualizes the data before and after clustering with Matplotlib/Seaborn. Outputs include cluster sizes, final centroid 
coordinates, and a scatter plot showing both clusters and centroids.

Input: city_data_50.csv with at least latitude and longitude columns.
Run: pip install pandas matplotlib seaborn → place the CSV next to the script → kmeans.py.

**COVID-19 Vaccination Impact Analysis**

A course project exploring how Ontario’s COVID-19 outcomes changed through the vaccine rollout. Using public Ontario Open Data, 
the notebook cleans and aggregates case, hospitalization/ICU, and mortality datasets, visualizes monthly trends, and 
aligns them with key vaccination milestones to show how severe outcomes declined post-rollout (correlation, not causation). 
Built with pandas and matplotlib.
