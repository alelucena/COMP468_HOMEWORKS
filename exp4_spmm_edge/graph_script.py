import random

M = 2000  # Number of nodes
Avg_Edges = 30

with open("graph_edges.txt", "w") as f:
    for u in range(M):
        # Give each node a random number of neighbors
        neighbors = random.sample(range(M), Avg_Edges)
        for v in neighbors:
            if u != v:
                f.write(f"{u} {v}\n")