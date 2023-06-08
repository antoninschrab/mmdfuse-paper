{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "b2292cdf-bf3e-41b5-9983-490b8adc2cce",
   "metadata": {
    "tags": []
   },
   "source": [
    "# Demo"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "78b01140-9fa7-40a8-a653-4cd516951c75",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "StreamExecutorGpuDevice(id=0, process_index=0, slice_index=0)"
      ]
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import numpy as np\n",
    "from mmdfuse import mmdfuse\n",
    "from jax import random\n",
    "import jax.numpy as jnp\n",
    "X = jnp.array([1, 2])\n",
    "X.device()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "765e39c4-96c1-4945-a655-49effa339e0d",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "# generate data\n",
    "key = random.PRNGKey(0)\n",
    "key, subkey = random.split(key)\n",
    "subkeys = random.split(subkey, num=2)\n",
    "X = random.uniform(subkeys[0], shape=(500, 10))\n",
    "Y = random.uniform(subkeys[1], shape=(500, 10)) + 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "6b44e490-e9f0-4dcb-9f75-c492ea06a39b",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "# compile function\n",
    "key, subkey = random.split(key)\n",
    "output, p_value = mmdfuse(X, Y, subkey, return_p_val=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "df9c4cdc-b341-4010-a2ca-dda01decd1c3",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "53.3 ms ± 45.7 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)\n"
     ]
    }
   ],
   "source": [
    "# Now the function runs fast for any inputs X and Y of the compiled shaped (500, 10)\n",
    "# If the shape is changed, the function will need to be compiled again\n",
    "key = random.PRNGKey(1) # different initialisation\n",
    "key, subkey = random.split(key)\n",
    "subkeys = random.split(subkey, num=2)\n",
    "X = random.uniform(subkeys[0], shape=(500, 10))\n",
    "Y = random.uniform(subkeys[1], shape=(500, 10)) + 1\n",
    "# see section below for detailed speed comparision between jax cpu and jax gpu \n",
    "%timeit output, p_value = mmdfuse(X, Y, subkey, return_p_val=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "fb8f1ef1-1867-411f-bf2d-9fe33aab26eb",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "output, p_value = mmdfuse(X, Y, subkey, return_p_val=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "47060fa1-2e7c-44a0-822b-f7311cfa0108",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Array(1, dtype=int32)"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# output is a jax array consisting of either 0 or 1\n",
    "output"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "e0e8afd9-72af-4972-ac0d-361f05d44040",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# to convert it to an int use: \n",
    "output.item()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "9b347c8c-10e4-408c-901b-7e276aedfe2e",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Array(0.00049975, dtype=float32)"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# p-value is a jax array\n",
    "p_value"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "a669b913-c278-44eb-9674-427706eed214",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.0004997501382604241"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# to convert it to an int use: \n",
    "p_value.item()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0cd15687-8197-42f7-a6fb-683850fdf198",
   "metadata": {
    "tags": []
   },
   "source": [
    "# Speed comparison"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "02919422-ca76-40ec-a2dc-fc2faa0270f8",
   "metadata": {
    "tags": []
   },
   "source": [
    "## Imports"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "44870f4b-5f30-4621-a4ab-6c0160f37ed6",
   "metadata": {},
   "source": [
    "Run only one of three next cells depending on whether to use Jax CPU or Jax GPU."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "df1a4636-78b3-4bc0-be50-778c60c093ea",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2023-05-23 14:30:48.087927: E external/org_tensorflow/tensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:266] failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected\n",
      "No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "CpuDevice(id=0)"
      ]
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# run for Jax CPU\n",
    "import os\n",
    "os.environ['CUDA_VISIBLE_DEVICES'] = ''\n",
    "import numpy as np\n",
    "from mmdfuse import mmdfuse\n",
    "from jax import random\n",
    "import jax.numpy as jnp\n",
    "%load_ext autoreload\n",
    "%autoreload 2\n",
    "X = jnp.array([1, 2])\n",
    "X.device()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "2a17e174-747c-4249-a950-75442de4fcc8",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "StreamExecutorGpuDevice(id=0, process_index=0, slice_index=0)"
      ]
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# run for Jax GPU\n",
    "import numpy as np\n",
    "from mmdfuse import mmdfuse\n",
    "from jax import random\n",
    "import jax.numpy as jnp\n",
    "%load_ext autoreload\n",
    "%autoreload 2\n",
    "X = jnp.array([1, 2])\n",
    "X.device()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1f18c90f-d430-4b92-bde9-a98873e41775",
   "metadata": {
    "tags": []
   },
   "source": [
    "## MMD-FUSE Runtimes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "78758346-9f95-4f80-81b4-5cbd7fdb903c",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "key = random.PRNGKey(0)\n",
    "key, subkey = random.split(key)\n",
    "subkeys = random.split(subkey, num=2)\n",
    "X = random.uniform(subkeys[0], shape=(500, 10))\n",
    "Y = random.uniform(subkeys[1], shape=(500, 10)) + 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "5cbbce66-3028-462a-83f9-1eebf2b667e3",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "# run for Jax CPU and Jax GPU to compile the function\n",
    "# Do not run for Numpy CPU\n",
    "output, p_value = mmdfuse(X, Y, subkey, return_p_val=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "8dc6549a-3def-4465-bb8e-527da50fbce2",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2.95 s ± 105 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)\n"
     ]
    }
   ],
   "source": [
    "# Jax CPU\n",
    "%timeit mmdfuse(X, Y, subkey, return_p_val=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "3a7a01d6-0076-46ec-accf-a3921d7d9ea3",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "54 ms ± 23.7 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)\n"
     ]
    }
   ],
   "source": [
    "# Jax GPU\n",
    "%timeit mmdfuse(X, Y, subkey, return_p_val=True)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.16"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
