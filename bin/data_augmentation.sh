#!/bin/bash

conda activate 8803 && cd ~/repo_team1 || exit
tmux new -d -s aug_fr 'python3 -m data_augmentation --gpu cuda:1 --language fr --limit 100'
tmux new -d -s aug_es 'python3 -m data_augmentation --gpu cuda:2 --language es --limit 100'
tmux new -d -s aug_de 'python3 -m data_augmentation --gpu cuda:3 --language de --limit 100'
tmux new -d -s aug_it 'python3 -m data_augmentation --gpu cuda:4 --language it --limit 100'