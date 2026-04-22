#!/bin/bash
# Run this once on Perlmutter to download all graph data
# Run with bash download_graphs.sh

mkdir -p $SCRATCH/graphs
cd $SCRATCH/graphs

echo "Downloading roadNet-CA..."
wget -nc https://snap.stanford.edu/data/roadNet-CA.txt.gz
gunzip -f roadNet-CA.txt.gz

echo "Downloading com-LiveJournal..."
wget -nc https://snap.stanford.edu/data/bigdata/communities/com-lj.ungraph.txt.gz
gunzip -f com-lj.ungraph.txt.gz

echo "Downloading ca-GrQc (small debug graph)..."
wget -nc https://snap.stanford.edu/data/ca-GrQc.txt.gz
gunzip -f ca-GrQc.txt.gz

echo "Done. Graphs are in $SCRATCH/graphs/"
ls -lh $SCRATCH/graphs/