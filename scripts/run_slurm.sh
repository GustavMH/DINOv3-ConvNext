#!/usr/bin/env bash

EPOCHS=100 bash scripts/run_pilot.sh distill sat tiny
EPOCHS=100 bash scripts/run_pilot.sh distill sat small
EPOCHS=100 bash scripts/run_pilot.sh distill sat base
EPOCHS=100 bash scripts/run_pilot.sh distill sat large

bash scripts/run_pilot.sh probe
