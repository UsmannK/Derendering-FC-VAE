#!/bin/bash
gsutil -m rsync -x ".*full.*" -d gs://quickdraw_dataset/sketchrnn $1
