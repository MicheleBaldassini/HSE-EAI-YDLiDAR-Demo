#!/bin/bash

# --- Variabili ---
PORT="/dev/ttyUSB0"
PYTHON_SCRIPT="/media/sf_VirtualBoxShared/EAI-YDLIDAR/src/gui.py"
VENV_DIR="/home/michele/Scrivania/venv"

# --- Controlla permessi della porta ---
PERMS=$(stat -c "%a" $PORT)
if [ "$PERMS" != "777" ]; then
    sudo chmod 777 $PORT
fi

source "$VENV_DIR/bin/activate"

python "$PYTHON_SCRIPT"