import os
inp_file = 'E_R'

with open('{}.txt'.format(inp_file),'r') as f:
    lines = f.readlines()

with open('{}_flow.txt'.format(inp_file),'w') as f:
    for l in lines:
        t0, t1, t2, _ = l.strip().split(' ')
        flow = t1.replace('/data/','/data/fw/')
        flow = flow.replace('.jpg','.flo')
        if os.path.exists('/media/filippo/Filippo/ComputerVision/Dataset/SelFlowProxy_Eigen/'+flow):
            f.write('{} {} {}\n'.format(t1, t2, flow))