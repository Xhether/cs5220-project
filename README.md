### Before starting

module load PrgEnv-gnu
module load cmake

mkdir build
cd build
cmake ..

### **To compile**

run the following in the build directory

```python
make
```

## To download graphs

```python
bash download_graphs.sh
```

**Running on small graph with no weights**

1. `./build/serial_reference --graph test_graphs/small_graph.txt --source 0 --unweighted --output results.txt`

**Running verify.py**

1. `python3 scripts/verify.py results.txt results.txt`

**mCreate interactive session**

`salloc --nodes 2 --qos interactive --time 01:00:00 --constraint cpu --account=m4341`
