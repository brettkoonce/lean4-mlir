#!/usr/bin/env bash
# The 3D UNet's second seed (plan §4d's open item): unet3d_20k.sh at --seed 1 under its own prefix. usage: unet3d_20k_s1.sh <gpu> [--resume …]
NAME=unet3d_20k_s1 exec "$(dirname "$0")/unet3d_20k.sh" "$1" --seed 1 "${@:2}"
