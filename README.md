### Before starting

module load PrgEnv-gnu
module load cmake

cd build

run cmake ..

### **To compile**

run the following in the build directory

```python
make
```

## To download graphs

```python
bash download_graphs.sh
```

**mCreate interactive session**

`salloc --nodes 2 --qos interactive --time 01:00:00 --constraint cpu --account=m4341`
