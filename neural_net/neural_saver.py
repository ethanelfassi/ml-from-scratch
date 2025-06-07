from .network import Network
import pickle

def save_network(file_name:str, network:Network):
    with open(file_name, "wb") as f:
        pickle.dump(network, f)

def load_network(file_name:str):
    with open(file_name, "rb") as f:
        network = pickle.load(f)
    return network