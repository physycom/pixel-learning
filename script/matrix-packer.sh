#! /usr/bin/env bash

output="merge.json"
{
  echo -n '['
  for j in output_json/*.json; do
#  for j in output_json/01**140054-93847*json; do
    content=$(cat $j)
    content=${content:1:${#content}-2}
    echo -n $content','
  done
  echo -n ']'
} > $output
sed 's/,]/]/g' $output > $output".tmp"
