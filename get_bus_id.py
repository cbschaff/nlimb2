import subprocess as sp

bus_ids = []
data = sp.check_output(['nvidia-smi', '-q']).decode()
for line in data.split('\n'):
    if 'Bus    ' in line:
        bus_ids.append(int(line.split(':')[-1].strip(), 16))

print(f'PCI:{bus_ids[0]}:0:0')
