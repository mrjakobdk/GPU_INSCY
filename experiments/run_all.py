import os


os.system('python experiments/run.py GPU-INSCY n 3 4 order')
os.system('python experiments/run.py INSCY n 3 4 order')

os.system('python experiments/run.py GPU-INSCY d 3 4')
os.system('python experiments/run.py INSCY d 3 4')

os.system('python experiments/run.py GPU-INSCY c 3 4')
os.system('python experiments/run.py INSCY c 3 4')

os.system('python experiments/run.py GPU-INSCY N_size 3 4')
os.system('python experiments/run.py INSCY N_size 3 4')

os.system('python experiments/run.py GPU-INSCY F 3 4')
os.system('python experiments/run.py INSCY F 3 4')

os.system('python experiments/run.py GPU-INSCY r 3 4')
os.system('python experiments/run.py INSCY r 3 4')

os.system('python experiments/run.py GPU-INSCY num_obj 3 4')
os.system('python experiments/run.py INSCY num_obj 3 4')

os.system('python experiments/run.py GPU-INSCY min_size 3 4')
os.system('python experiments/run.py INSCY min_size 3 4')