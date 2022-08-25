from utils.telegram import send_telegram
import subprocess

gpu_type = "gpu" # gpupro, gpua100

send_telegram(f"§§§ src/get_gpu.py started\nrequesting: {gpu_type}")

bashCommand = f"sblock -p {gpu_type} --gpus=1 -c 8 --mem=64gb --time=70:00:00"
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

send_telegram(f"§§§ src/get_gpu.py finished\ngot: {gpu_type}")