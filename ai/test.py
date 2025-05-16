#!/usr/bin/env python3
from hailo_platform.pyhailort.pyhailort import Device, HEF, InputVStreams, OutputVStreams

# 1) connect to the chip
dev = Device()

# 2) load your HEF
hef = HEF("/home/pi/models/yolov8n.hef")

# 3) configure it (returns a list of network-groups; normally 1)
ngs = dev.configure(hef)
ng = ngs[0]

# 4) set up the I/O streams
in_params  = ng.get_input_vstream_infos()[0]
out_params = ng.get_output_vstream_infos()[0]

with InputVStreams(ng, {in_params.name: in_params}) as ins, \
     OutputVStreams(ng, {out_params.name: out_params}) as outs:

    # build a dummy input
    import numpy as np
    dummy = np.zeros(ins.get().shape, dtype=ins.get().dtype)

    # send it and receive
    ins.get().send(dummy)
    dets = outs.get().recv()
    print("Detection tensor(s):", dets)
