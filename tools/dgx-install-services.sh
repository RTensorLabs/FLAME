#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/kenpath/flame-training/FLAME
mkdir -p "$HOME/.config/systemd/user"
cat > "$HOME/.config/systemd/user/flame-30m.service" <<'EOF'
[Unit]
Description=FLAME 30M two-stage DGX training with recovery snapshots
After=network-online.target
StartLimitIntervalSec=0
[Service]
Type=simple
WorkingDirectory=/home/kenpath/flame-training/FLAME
Environment=CARGO_HOME=/home/kenpath/flame-training/.cargo
Environment=RUSTUP_HOME=/home/kenpath/flame-training/.rustup
Environment=PATH=/home/kenpath/flame-training/.cargo/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
Environment=LD_LIBRARY_PATH=/usr/local/cuda/lib64
Environment=RAYON_NUM_THREADS=4
Environment=PYTHONUNBUFFERED=1
ExecStart=/home/kenpath/flame-training/FLAME/.venv/bin/python tools/dgx_train.py run
Restart=on-failure
RestartSec=120
MemoryMax=20G
CPUQuota=400%
Nice=10
TimeoutStopSec=300
StandardOutput=append:/home/kenpath/flame-training/FLAME/ops/runner.log
StandardError=append:/home/kenpath/flame-training/FLAME/ops/runner.log
UMask=0077
[Install]
WantedBy=default.target
EOF
cat > "$HOME/.config/systemd/user/flame-checkpoint-publisher.service" <<'EOF'
[Unit]
Description=Publish FLAME recovery checkpoints and completed model to GitHub
[Service]
Type=oneshot
WorkingDirectory=/home/kenpath/flame-training/FLAME
ExecStart=/home/kenpath/flame-training/FLAME/.venv/bin/python tools/dgx_train.py publish
Environment=PYTHONUNBUFFERED=1
MemoryMax=1G
Nice=15
TimeoutStartSec=7200
StandardOutput=append:/home/kenpath/flame-training/FLAME/ops/publisher.log
StandardError=append:/home/kenpath/flame-training/FLAME/ops/publisher.log
UMask=0077
EOF
cat > "$HOME/.config/systemd/user/flame-checkpoint-publisher.timer" <<'EOF'
[Unit]
Description=Retry FLAME checkpoint publishing every five minutes
[Timer]
OnBootSec=60
OnUnitInactiveSec=300
Persistent=true
Unit=flame-checkpoint-publisher.service
[Install]
WantedBy=timers.target
EOF
systemctl --user daemon-reload
systemctl --user enable --now flame-checkpoint-publisher.timer flame-30m.service
