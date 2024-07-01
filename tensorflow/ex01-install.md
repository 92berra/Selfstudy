# M3 Macbook Pro Tensorflow Install Guide

<br/>

## Environment Setting

```
conda create -n tf-metal python=3.8
conda activate tf-metal
```

<br/>

## Install Tensorflow Dependencies

```
conda install -c apple tensorflow-deps
```

<br/>

## Install Tensorflow

```
python -m pip install tensorflow-macos
```

<br/>

## Install Tensorflow Metal

To utilize the Apple's gpu framework for M3, M3 Pro and M3 Max.

```
python -m pip install tensorflow-metal
```

<br/>

## Install Jupyter

```
conda install -c conda-forge jupyter
```

<br/>

## Varification

```
ipython
```

```
In [1]: import tensorflow as tf

In [2]: tf.__version__

In [3]: from tensorflow.python.client import device_lib

In [4]: device_lib.list_local_devices()
```

```
2024-07-01 21:13:32.032148: I metal_plugin/src/device/metal_device.cc:1154] Metal device set to: Apple M3
2024-07-01 21:13:32.032223: I metal_plugin/src/device/metal_device.cc:296] systemMemory: 16.00 GB
2024-07-01 21:13:32.032239: I metal_plugin/src/device/metal_device.cc:313] maxCacheSize: 5.33 GB
2024-07-01 21:13:32.032401: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:303] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
2024-07-01 21:13:32.032491: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:269] Created TensorFlow device (/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)
Out[4]: 
[name: "/device:CPU:0"
 device_type: "CPU"
 memory_limit: 268435456
 locality {
 }
 incarnation: 10637349104929960166
 xla_global_id: -1,
 name: "/device:GPU:0"
 device_type: "GPU"
 locality {
   bus_id: 1
 }
 incarnation: 4220327561894979199
 physical_device_desc: "device: 0, name: METAL, pci bus id: <undefined>"
 xla_global_id: -1]
```

Congratulations. 🎉


<br/>
<br/>
<br/>
<br/>

<div align='center'>
92berra ©2024
</div>