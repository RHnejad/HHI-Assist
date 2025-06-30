#!/bin/bash

input_folder="."

for bvh_file in "$input_folder"/*.bvh; do

	"bvh-converter" "$bvh_file"

done

