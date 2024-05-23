import time
import pickle as pkl

start_t = time.time()

with open('/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/big_replay.pkl', 'rb') as f:
    data = pkl.load(f)

end_t = time.time()

print("File loaded in time: ", (end_t - start_t) / 60.)