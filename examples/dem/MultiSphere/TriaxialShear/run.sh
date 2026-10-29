#!/usr/bin/env bash
cd "$(dirname "$0")"

python conso.py
python drained.py
python undrained.py
