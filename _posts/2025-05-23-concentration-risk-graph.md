---
layout: post
title: Concentration Risk via Graph ML
---

## Untangling Counterparty Risk: A Graph ML Approach to Simulating Systemic Shocks

This project has been a journey. What started as an interest in financial stability and the cascading effects seen in economic crises quickly evolved into a deep dive into graph machine learning. The goal was to model counterparty networks to assess concentration risk and, more ambitiously, to simulate how systemic shocks might propagate through such a network. The main tools for this endeavor became NetworkX for the foundational graph work and PyTorch Geometric for the simulation aspects.

### The Problem: Concentration and Contagion

The core idea revolves around concentration risk – the risk that too much exposure is concentrated with a few counterparties. If one of these key entities faces distress, the ripple effects can be substantial. Traditional methods often look at direct exposures, but the interconnectedness of financial systems means indirect exposures and network effects are critically important. This is where graph theory felt like a natural fit. My initial thought was, "How can I represent these complex relationships and then stress-test them?"

### Building the Network: From Theory to NetworkX

First, I needed a network. Given the difficulty in obtaining real, granular counterparty exposure data for a student project, I decided to generate a synthetic dataset. I focused on creating a plausible structure, aiming for something that might resemble a simplified interbank network. I made assumptions: some nodes would be highly connected (major banks), while others would have fewer connections (smaller institutions).

I used NetworkX to construct the graph. Each node represents a financial institution, and each directed edge `(u, v)` represents an exposure of `u` to `v`, with the weight of the edge signifying the monetary value of that exposure.

```python
import networkx as nx
import random

num_institutions = 50
# Create a scale-free graph as a base - seemed like a reasonable starting point
# for financial networks based on some literature.
G = nx.barabasi_albert_graph(num_institutions, m=3, seed=42)

# Add exposure values as edge attributes
for u, v in G.edges():
    # Exposures are not necessarily symmetric
    G.edges[u,v]['exposure'] = random.uniform(1e6, 1e8) # u is exposed to v

# Also need to consider assets for each institution for shock absorption
for node in G.nodes():
    # Let's assume total assets are some multiple of their outgoing exposures, plus a base
    total_outgoing_exposure = sum(G.edges[node, successor]['exposure'] for successor in G.successors(node) if 'exposure' in G.edges[node, successor])
    G.nodes[node]['initial_capital'] = random.uniform(0.05, 0.15) * total_outgoing_exposure + random.uniform(1e7, 5e7) # Capital buffer
    G.nodes[node]['status'] = 'solvent' # Initial status
```

One of the early challenges was deciding how to represent "capital" or "assets" for each institution. This was crucial for the shock simulation later, as it would determine an institution's capacity to absorb losses. I settled on a simplified model where initial capital was related to an institution's total outgoing exposures plus some base amount, ensuring larger players had more capital but also more at stake.

### Identifying Concentration: Basic Graph Metrics

With the graph built, assessing concentration risk was the next step. Initially, I looked at simple metrics like degree centrality – which nodes have the most connections. However, raw degree doesn't account for the *value* of those connections. So, weighted degree (sum of exposures) became more relevant.

```python
# Calculate weighted degree (sum of outgoing exposures for each node)
weighted_degrees = {}
for node in G.nodes():
    outgoing_exposure = sum(G.edges[node, successor]['exposure'] for successor in G.successors(node) if 'exposure' in G.edges[node, successor])
    weighted_degrees[node] = outgoing_exposure

# Identify top N concentrated institutions
sorted_by_exposure = sorted(weighted_degrees.items(), key=lambda item: item, reverse=True)
# print("Top 5 institutions by total outgoing exposure:")
# for i in range(min(5, len(sorted_by_exposure))):
# print(f"Institution {sorted_by_exposure[i]}: {sorted_by_exposure[i]:.2f}")
```
This gave a basic idea of where direct risk was concentrated. But I wanted to simulate the secondary effects – the contagion.

### Stepping into Systemic Shock Simulation with PyTorch Geometric

This was the part I was most excited and nervous about. How do you model a shock wave? I knew NetworkX alone would be cumbersome for iterative updates across the graph, especially if I wanted to incorporate any learning aspects later (though I didn't get to that in this iteration). This led me to PyTorch Geometric (PyG). My reasoning was that PyG is designed for graph-based machine learning and has efficient ways to handle graph data and message passing, which seemed conceptually similar to how a shock might propagate.

The first hurdle was converting my NetworkX graph into a PyG `Data` object. This took a bit of fiddling, especially ensuring edge attributes (the exposure values) were correctly mapped.

```python
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

# Need to ensure nodes are indexed from 0 to num_nodes - 1 for PyG
# My G generated by barabasi_albert_graph already has this property.

# Extract edge indices and attributes
edge_index_list = []
edge_attr_list = [] # This will store exposures

for u, v, data in G.edges(data=True):
    edge_index_list.append([u,v])
    edge_attr_list.append(data['exposure'])

edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
edge_attr = torch.tensor(edge_attr_list, dtype=torch.float).unsqueeze(1) # Make it [num_edges, 1]

# Node features: for now, let's use initial capital.
# Status ('solvent', 'stressed', 'failed') will be managed separately or encoded if using a learning model.
x_list = [G.nodes[node]['initial_capital'] for node in sorted(G.nodes())]
x = torch.tensor(x_list, dtype=torch.float).unsqueeze(1) # [num_nodes, 1]

pyg_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
# print(pyg_data)
```
I spent a good afternoon debugging why `edge_attr` wasn't aligning with `edge_index`. It turned out to be a combination of graph directionality and the order in which NetworkX was returning edges versus what PyG expected. A few print statements and careful re-reading of the `from_networkx` documentation and related StackOverflow posts eventually cleared that up. The key was to be meticulous about the source and target nodes for directed edges when building `edge_index` and ensuring `edge_attr` followed the exact same ordering.

### Simulating the Shock: A Custom Propagation Logic

I didn't use a pre-built GNN model like GCN or GAT for *predicting* failures in this stage. Instead, I used PyG's structure to facilitate a rule-based simulation. The idea was:
1.  Introduce a shock: Select one or more institutions to fail initially. This means their capital drops to zero.
2.  Propagation: For each institution `j` that has an exposure to a failed institution `i`, `j` incurs a loss.
3.  Update status: If `j`'s losses exceed its capital, it also fails.
4.  Iterate: Repeat steps 2 and 3 until no more failures occur in an iteration.

I implemented this as a loop. Inside the loop, I'd identify newly failed nodes and then calculate the impact on their creditors.

```python
# Simulation parameters
initial_shock_node = 0 # Let's say institution 0 fails
capital_threshold_for_failure = 0 # If capital drops to or below this, it fails

# Make a copy of initial capitals to modify during simulation
current_capital = pyg_data.x.clone().squeeze() # Work with a 1D tensor for capital
node_status = {node_id: 'solvent' for node_id in G.nodes()}

# Initial shock
current_capital[initial_shock_node] = 0
node_status[initial_shock_node] = 'failed'
failed_in_current_step = {initial_shock_node}
all_failed_nodes = {initial_shock_node}

# print(f"Initial shock: Node {initial_shock_node} fails.")

# Simulation loop
for step in range(num_institutions): # Max iterations to prevent infinite loops
    if not failed_in_current_step:
        # print(f"No new failures in step {step}. Simulation stable.")
        break

    newly_failed_this_iteration = set()
    # Who is exposed to the nodes that just failed?
    # pyg_data.edge_index gives [source_nodes, target_nodes]
    # We want to find source nodes whose targets are in failed_in_current_step

    for creditor_idx in range(pyg_data.num_nodes):
        if node_status[creditor_idx] == 'failed':
            continue # Already failed, skip

        loss_for_this_creditor = 0
        # Iterate through all edges to find relevant exposures
        for i in range(pyg_data.edge_index.size(1)):
            source, target = pyg_data.edge_index[0, i].item(), pyg_data.edge_index[1, i].item()
            # If creditor_idx is the source and target is a node that failed in the *previous* step
            if source == creditor_idx and target in failed_in_current_step:
                exposure_value = pyg_data.edge_attr[i].item()
                loss_for_this_creditor += exposure_value
                # print(f"  Node {creditor_idx} incurs loss of {exposure_value} from failed node {target}")


        if loss_for_this_creditor > 0:
            current_capital[creditor_idx] -= loss_for_this_creditor
            # print(f"  Node {creditor_idx} new capital: {current_capital[creditor_idx]}")
            if current_capital[creditor_idx] <= capital_threshold_for_failure and node_status[creditor_idx] == 'solvent':
                node_status[creditor_idx] = 'failed'
                newly_failed_this_iteration.add(creditor_idx)
                # print(f"  Node {creditor_idx} has now failed due to losses.")


    failed_in_current_step = newly_failed_this_iteration
    all_failed_nodes.update(failed_in_current_step)

    # print(f"Step {step + 1}: Nodes {failed_in_current_step} failed. Total failed: {len(all_failed_nodes)}")

# print(f"\nSimulation finished. Total nodes failed: {len(all_failed_nodes)}")
# print(f"Failed nodes: {all_failed_nodes}")
```

One major point of confusion was how to efficiently update node states and propagate effects using PyG's tensor operations. My first approach for iterating through exposures was quite naive and slow, involving nested Python loops. I knew there had to be a more "graph-native" way, perhaps involving PyG's message passing classes if I were to formalize it more. For this iteration, I stuck to a semi-vectorized approach by iterating through nodes and then their relevant edges, but I can see how defining a custom `MessagePassing` class in PyG would be the more elegant and scalable solution for the propagation step. I actually tried to implement a custom `propagate` and `message` function but got bogged down in how to correctly aggregate messages representing losses. I found a forum post discussing a similar cascading failure model, which gave me some ideas, but ultimately, for this version, the explicit loop over creditors felt more transparent for debugging, given my timeline.

The `edge_index` in PyG stores edges as a `[2, num_edges]` tensor, where `edge_index[0]` are source nodes and `edge_index[1]` are target nodes. So, if bank `A` lends to bank `B`, the edge is `(A, B)`. If `B` fails, `A` suffers a loss. My `edge_attr` stores the exposure amount. So, in the simulation, when a node `target` fails, I needed to find all `source` nodes that had an edge pointing to `target` and reduce their capital by the corresponding `edge_attr`. This directionality was important and easy to get wrong.

### A Moment of "Aha!" and Frustration

A significant challenge was managing the state of nodes ('solvent', 'stressed', 'failed') and ensuring the simulation proceeded in logical steps. Initially, my simulation was causing nodes to fail and recover in the same step, or losses weren't cascading correctly because I was updating capitals "live" within an iteration, affecting calculations for other nodes in that same iteration. The "aha!" moment came when I realized I needed to buffer the "newly failed" nodes in each step and only apply their impact in the *next* step. This batching of failures per iteration made the simulation behave much more predictably.

Debugging the loss propagation logic was also tricky. `print` statements became my best friend. I'd often print out `current_capital`, the `failed_in_current_step` set, and the specific losses being calculated. There was one evening I spent hours because losses were being double-counted or not applied to the correct creditor due to an indexing error when accessing `edge_attr` relative to the filtered edges. It turned out I was iterating over `G.edges()` from the NetworkX object inside a loop that should have been using PyG's `edge_index` consistently. Switching to solely use `pyg_data.edge_index` and `pyg_data.edge_attr` within the core simulation loop resolved this.

### Results (So Far) and Reflections

The simulation, even in its current form, can show how the failure of one institution (especially a highly connected or largely exposed one) can lead to a cascade of failures. By varying the initially shocked node and its capital, I could observe different extents of contagion. For instance, shocking a node with high weighted degree centrality but low capital might not be as devastating as shocking a node with slightly lower degree but whose failure impacts counterparties that are themselves systemically important or undercapitalized.

This project is far from over. The current simulation is rule-based. A natural next step would be to explore actual GNN models. Could a GCN or a GraphSAGE model learn to *predict* the likelihood of a node failing given the state of its neighbors and its own features, perhaps trained on historical (simulated) crisis data? That's a much larger undertaking.

I also considered using attention mechanisms (like in GATs) for the propagation, where the "attention" could represent how much impact a failing neighbor has, potentially learned from data rather than being a direct function of exposure size alone. But this felt like a step too far for the current scope, given I first needed to get the basic simulation mechanics right.

Time constraints were a big factor. Properly tuning the synthetic data generation, exploring more sophisticated graph metrics for concentration (like PageRank or community detection to find clusters of risk), and implementing a more robust PyG `MessagePassing` framework are all areas for future work. I also only scratched the surface of visualizing these dynamic graphs; an animation of the shock propagation would be very insightful.

Overall, this project was a fantastic learning experience in applying graph theory and the very basics of graph-based simulation to a complex real-world problem. It highlighted the power of representing systems as networks and gave me a new appreciation for the intricacies of financial stability. Wrestling with NetworkX and PyTorch Geometric has definitely leveled up my technical skills, even if there were moments of pure frustration staring at non-converging simulations or mismatched tensor dimensions.