"""
Network topology module - provides network and layer classes.
"""
from network.topology.network import Network
from network.topology.sequential_network import SequentialNetwork
from network.topology.neuron import NeuronLayer
from network.topology.connection import Connection
from network.topology.simple_connection import SimpleConnection

__all__ = [
    "Network", "SequentialNetwork", 
    "NeuronLayer", 
    "Connection", "SimpleConnection",
]

