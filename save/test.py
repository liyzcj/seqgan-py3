import pickle
import numpy

filename = 'save/target_params.pkl'

with open(filename, 'rb') as f:
    d = pickle.load(f, encoding='bytes') 

print(d)