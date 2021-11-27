#!/usr/bin/env bash

tmux new-session -d -s augmentation_fr 'python3 -m data_augmentation --language fr --gpu cuda:1'
tmux new-session -d -s augmentation_es 'python3 -m data_augmentation --language es --gpu cuda:2'
tmux new-session -d -s augmentation_de 'python3 -m data_augmentation --language de --gpu cuda:3'
tmux new-session -d -s augmentation_it 'python3 -m data_augmentation --language it --gpu cuda:4'