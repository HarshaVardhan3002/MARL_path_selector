# network_components.py

class Resource:
    """Represents a resource (e.g., a router or link) in the network."""
    def __init__(self, id, capacity):
        self.id = id
        self.capacity = capacity  # This is the C_j from the paper
        
        # This is for the dual algorithm, the resource price mu_j
        self.mu = 0.1 # Start with a small non-zero price

class Path:
    """Represents a route or path from a source to a destination."""
    def __init__(self, id, source_id, resources, rtt):
        self.id = id
        self.source_id = source_id # Which source this path belongs to
        # A list of Resource objects that this path uses
        self.resources = resources 
        self.rtt = rtt # This is the T_r from the paper

        # This is the variable for the flow rate x_r(t)
        self.x = 1.0 # Start with a flow of 1.0

class Source:
    """Represents a source of traffic."""
    def __init__(self, id):
        self.id = id
        
        # This is for the dual algorithm, the total source rate y_s(t)
        self.y = 1.0 # Start with a rate of 1.0
