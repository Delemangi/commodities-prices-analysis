from sys import argv

import pandas as pd
from jaal import Jaal

nodes_list = ["Crude Oil", "Natural Gas", "Gold", "S&P500", "BSE", "FTSE", "Hang Seng", "Dow Jones"]

nodes = pd.DataFrame({"name": nodes_list, "id": nodes_list})

edges = pd.read_csv(argv[1])
edges.rename(columns={"A": "from", "B": "to"}, inplace=True)
edges["Importance"] = edges["Importance"].apply(abs)
Jaal(edges, nodes).plot(directed=True)
