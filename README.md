# Minecraft
This repo contains code to collect random trajectories in Minecraft

# Installation

```
sudo apt install xvfb
pip install -r requirements.txt
```

# Collect Trajectories

`sh xvfb_run.sh python collect.py -o data -z 32`

*Note: Sometimes the MineRL environments for some workers may crash / timeout meaning that you will not collect the full number of specified episodes*
