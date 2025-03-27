from tqdm import tqdm
import time

pbar = tqdm(total=100, desc="Processing", ncols=100, leave=False)
for i in range(100):
    time.sleep(0.1)
    pbar.update()
pbar.close()

for i in tqdm(range(100), desc="Working", ncols=100):
    time.sleep(0.1)
