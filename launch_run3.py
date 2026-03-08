import paramiko
import time

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect("10.82.34.5", port=15654, username="shenxiang", password="@lhpcsoi#510-2", timeout=15)

sftp = ssh.open_sftp()
script_lines = [
    "#!/bin/bash\n",
    "cd /data/Lab105/huangjiapeng/bendi\n",
    "PYBIN=/home/shenxiang/miniconda3/envs/huangjiapeng/bin/python\n",
    "LOG=results/train_logs/run3_cremad_balanced_$(date +%Y%m%d_%H%M%S).log\n",
    "echo \"Logging to: $LOG\"\n",
    "nohup $PYBIN main.py \\\n",
    "    --train \\\n",
    "    --dataset CREMAD \\\n",
    "    --fusion_method concat \\\n",
    "    --batch_size 64 \\\n",
    "    --epochs 70 \\\n",
    "    --optimizer sgd \\\n",
    "    --learning_rate 0.002 \\\n",
    "    --lr_decay_step 40 \\\n",
    "    --lr_decay_ratio 0.1 \\\n",
    "    --gpu_ids 1 \\\n",
    "    --use_cmob \\\n",
    "    --use_sample_weighting \\\n",
    "    --use_cross_gate \\\n",
    "    --lambda_audio 1.2 \\\n",
    "    --lambda_visual 1.35 \\\n",
    "    --cross_gate_strength_audio 1.1 \\\n",
    "    --cross_gate_strength_visual 1.2 \\\n",
    "    --cmob_beta_cap 1.2 \\\n",
    "    --cmob_beta_warmup_epochs 5 \\\n",
    "    --cmob_ema_momentum 0.9 \\\n",
    "    --cmob_visual_boost 0.3 \\\n",
    "    --focal_gamma_end 2.0 \\\n",
    "    --sw_k_threshold 0.05 \\\n",
    "    > $LOG 2>&1 &\n",
    "echo \"PID: $!\"\n",
    "echo $! > /tmp/run3_pid.txt\n",
]
with sftp.file("/tmp/run3.sh", "w") as f:
    f.writelines(script_lines)
sftp.close()

stdin, stdout, stderr = ssh.exec_command("bash /tmp/run3.sh")
time.sleep(5)
out = stdout.read().decode()
err = stderr.read().decode()
print("OUT:", out)
if err:
    print("ERR:", err[:800])

time.sleep(2)
stdin2, stdout2, _ = ssh.exec_command("cat /tmp/run3_pid.txt 2>/dev/null; echo ---PROCS---; ps aux | grep main.py | grep -v grep")
print("Process check:", stdout2.read().decode())

ssh.close()
print("Done.")
