# Deep Learning Compiler Fuzzing

## Group Members

- Mohammed Toufikuzzaman
- Mujtahid Al-Islam Akon

## Milestone 1

### Implementation of Baseline Work

**Baseline Paper:** WhiteFox: White-Box Compiler Fuzzing Empowered by Large Language Models

## Overview of the Problem

Deep learning models are developed using high-level frameworks like PyTorch and TensorFlow. While training and prototyping these models in these high-level frameworks provide flexibility, deploying these models requires performance improvements. These performance improvements and optimizations are performed by deep learning compilers like PyTorch Inductor, TensorFlow XLA, TensorFlow Lite, and GLOW. These compilers take computational graphs as input and produce highly optimized target binaries for a versatile set of target devices, including CPUs, GPUs, TPUs, and a variety of edge devices. This process is quite complex and has the potential to introduce bugs.

WhiteFox is the latest work that performs LLM-guided fuzzing on deep learning compilers. In this document, we provide a comprehensive overview of how to set up and run the fuzzing process of WhiteFox and also offer a brief explanation of a few identified bugs in PyTorch and TensorFlow compilers.

## How WhiteFox Works

The input of the fuzzing process is ML models written in high-level frameworks like PyTorch and TensorFlow. The fuzzing process is prepared in three steps:

1. First, the authors identified around 50 optimization-related functions in the source code of these compilers. They designed prompts that are sent to the GPT-4 model to analyze the source code of the optimizations and generate high-level instructions or pseudocode on how to design an ML model to trigger a specific optimization.
2. The output from GPT-4 is then forwarded to another LLM, StarCoder, with a prompt to generate an ML model in PyTorch or TensorFlow using the instructions from GPT-4.
3. Finally, the output from StarCoder is executed with optimizations turned off and on to validate the triggering of any vulnerability.

## Running the Fuzzer

### Prerequisite

1. Clone the testing repository
    - `git clone https://github.com/tzpranto/WhiteFox_Test`
2. Set up your environment with the CUDA toolkit. We recommend using Anaconda for environment management. You can install the CUDA toolkit with the following command:
    - `conda install nvidia::cuda-toolkit`
3. Python version >= 3.9.0 (It must support f-strings.)
   - Highly recommended to use Python 3.9.
4. Check our dependent Python libraries in `requirements.txt` and install them with:
   - `pip install -r requirements.txt`
5. Install StarCoder:
   - Please follow the instructions in [StarCoder](https://huggingface.co/bigcode/starcoder).

### Running WhiteFox

#### Step 1: Request Summarization

The prompts for NL generation are in [Prompts](Prompts) with the format `Prompts/{compiler}/src2req/{name}.txt`.

If you want to generate the prompt by yourself, take the prompt for `torch-inductor` as an example:

```bash
bash scripts/whitefox-torch-prompt-gen-src2req.sh
# Or
bash scripts/whitefox-torch-prompt-gen-src2req.sh {generated-prompt-dir}
```

The generated prompts will be in `Prompts-generated` by default.

After getting the prompts, you can run the following command to generate the requirements:

```bash
python gpt4.py --prompt-dir=Prompts/torch-inductor/src2req \ 
    --outdir=Requirements/torch-inductor/req \
    --temperature=0.0 \
    --batch-size=1
```

Before running the command, please put your OpenAI API key in `openai.key`:

```bash
echo {api_key} > openai.key
```

#### Step 2: Test Generation

First, you need to generate the prompts for the test generation based on the requirements:

```bash
bash scripts/whitefox-torch-prompt-gen-req2test.sh 

# Or
bash scripts/whitefox-torch-prompt-gen-req2test.sh {req-dir} {generated-prompt-dir}
```

The generated prompts will be in `Prompts-generated` by default.

Or you can use the prompts we generated in [Prompts](Prompts) with the format `Prompts/{compiler}/req2test/{name}.txt`.

We leverage [StarCoder](https://huggingface.co/bigcode/starcoder) to generate the tests based on the prompts.

##### [Option 1]: Local Mode (Recommended!)

We recommend using the local mode to generate the tests, which utilizes [vllm](https://github.com/vllm-project/vllm).

You can execute the following command to generate the tests:

```bash
python starcoder_gen.py --hf-home={path-to-huggingface} --hf-cache={path-to-huggingface-cache} --prompt-dir=Prompts/torch-inductor/req2test ----output-dir=starcoder-generated --num=10 
```

The generated tests will be in `starcoder-generated`.

##### [Option 2]: Server Mode

You can execute the following command to generate the tests:

1. Run the StarCoder server:

```bash
python starcoder_service.py --hf-home={path-to-huggingface} --hf-cache={path-to-huggingface-cache} --prompt-dir=starcoder-prompts --outdir=starcoder-generated --device='cuda:0' --num=10 --batch_size=10
```

2. Put the prompts in `starcoder-prompts`, and the generated tests will be in `starcoder-generated`.

```bash
mkdir starcoder-prompts/torch-inductor
cp -r Prompts/torch-inductor/req2test starcoder-prompts/torch-inductor/
```

#### Step 3: Test Execution

You can execute the following command to run the tests:

```bash
cd torch-exec && python run_torch.py --input-dir=../starcoder-generated/torch-inductor-generated/step1 --res-dir=_results-torch
```

The output of the execution will be in `torch-exec/_results-torch`.

## Evaluating the Detecting Vulnerabilities

In this section, we discuss three vulnerabilities from PyTorch and TensorFlow that were detected by the WhiteFox authors. They have detected about 96 bugs, and the list is available in the `bugs.csv` file of the testing repository mentioned above. Please note that they tested the fuzzing inputs on specific nightly versions of PyTorch and TensorFlow, which are no longer available. We tried our best to emulate their environment as far as possible. However, as expected, we could not reproduce all the detected bugs. We are providing samples of how the bugs can be reproduced, which can be tried for any bugs listed in `bugs.csv`.

### Prerequisite

1. Clone the testing repository
    - `git clone https://github.com/tzpranto/DL_Compiler_Fuzzer`
2. Set up your environment with the CUDA toolkit. We recommend using Anaconda for environment management. You can install the CUDA toolkit with the following command:
    - `conda create --name <env_name> python=3.9`
    - `conda install nvidia/label/cuda-11.8.0::cuda-toolkit`
    - `pip install torch==2.0.0+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
    - `pip install tensorflow==2.13.0`
3. Each repository has two folders `pytorch` and `tensorflow` with sample fuzzing input that may cause the compiler to crush or output tensors that are not same with the unoptimized version. Run a test by using the command
    - `python <test_case_filename>.py`
4. Follow the corresponding GitHub issue in `bugs.csv` file for more details

### PyTorch Vulnerabilities
#### Case 1: `torch.compile` raises `dense_to_mkldnn` expects `float` or `bfloat16` tensor input after doing some optimization. If we don't run it in `torch.no_grad` nor set train as `False`, it will succeed.


```python
import torch
import torch.nn as nn

torch.manual_seed(420)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(2, 2).to(torch.float64)
    
    def forward(self, x):
        x = x.permute(1, 0)
        x = self.linear(x)
        x = x.permute(1, 0)
        return x

input_tensor = torch.rand(2, 2).to(torch.float64)
func = Model().to('cpu')
print(func(input_tensor))
# tensor([[-1.0019, -0.4457],
#        [ 0.1512, -0.4111]], dtype=torch.float64, grad_fn=<PermuteBackward0>)

with torch.no_grad():
    func.train(False)
    jit_func = torch.compile(func)
    print(jit_func(input_tensor))
    # torch._dynamo.exc.BackendCompilerFailed: backend='debug_wrapper' raised:
    # RuntimeError: dense_to_mkldnn expects float or bfloat16 tensor input
```

#### Case 2: `torch.compile` doesn't support `permute(*tensor)`. It will just raise an `torch._dynamo.exc.TorchRuntimeError`.


```python
import torch
import torch.nn as nn

torch.manual_seed(420)

class MyModel(torch.nn.Module):

    def forward(self, input):
        permute = torch.tensor([0, 2, 1])
        x = input.permute(*permute)
        return x

input = torch.randn(2, 3, 4)

func = MyModel()
jit_func = torch.compile(func)

print(func(input))
# tensor([[[-0.0070,  0.0302,  0.5020],
#          [ 0.5044,  0.3826,  0.7538],
#          [ 0.6704, -0.5131,  0.6128],
#          [-0.3829,  0.7104, -0.9300]],
# 
#         [[-0.7392, -1.4816, -0.0021],
#          [ 0.4839,  0.3298, -1.1769],
#          [ 2.0201,  0.4856,  0.8036],
#          [-0.3333,  0.4131, -1.2524]]])

print(jit_func(input))
# torch._dynamo.exc.TorchRuntimeError
```

#### Case 3: `torch.compile` raises an error that expanded size doesn't match when enabling shape_padding by setting `TORCHINDUCTOR_SHAPE_PADDING=1`.


```python
import torch
import torch.nn as nn

torch.manual_seed(420)

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.functional.linear
        self.linear_weight = torch.randn(4, 4).cuda()
        self.bias = torch.randn(1, 4).cuda()

    def forward(self, x):
        x = self.linear(x, self.linear_weight, self.bias)
        return x

input_tensor = torch.randn(1, 3, 4).cuda()

func = Model().cuda()

res1 = func(input_tensor)
print(res1)
# tensor([[[-1.2507,  1.2743,  2.1668,  2.3092],
#          [ 0.2125,  0.0958, -2.3418,  3.3766],
#          [-0.3756,  0.8750, -0.5950,  4.4472]]], device='cuda:0')

jit_func = torch.compile(func)
res2 = jit_func(input_tensor)
# RuntimeError: The expanded size of the tensor (4) must match the existing size (2) at non-singleton dimension 0.  Target sizes: [4, 4].  Tensor sizes: [2, 4]
# While executing %linear : [#users=1] = call_function[target=torch._C._nn.linear](args = (%l_x_, %l__self___linear_weight, %l__self___bias), kwargs = {})
```


### Tensorflow Vulnerabilities
#### Case 1: Model produces wrong results.
    - Keras mode output:  tf.Tensor([[100. 100.]], shape=(1, 2), dtype=float32)
    - Lite mode (compiled) output:  [[inf inf]]


```python
import tensorflow as tf
import numpy as np

x1 = tf.constant([100., 100.], shape=[1, 2])

class Model(tf.keras.Model):

  def __init__(self):
    super(Model, self).__init__()

  @tf.function(input_signature=[tf.TensorSpec(x1.shape, x1.dtype)])
  def call(self, x):
    return tf.math.softplus(x)

# Initializing the model
m = Model()

m(x1)
print('Keras mode output: ', m(x1))

converter = tf.lite.TFLiteConverter.from_keras_model(m)
tflite_model = converter.convert()

def _evaluateTFLiteModel(tflite_model, input_data):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for i in range(len(input_data)):
        interpreter.set_tensor(input_details[i]['index'], input_data[i])

    interpreter.invoke()
    output_data = [interpreter.get_tensor(output_details[i]['index'])
                   for i in range(len(output_details))]
    return output_data

print('Lite mode output: ', _evaluateTFLiteModel(tflite_model,[x1])[0])
```

#### Case 2: Model produces wrong results.
    - Keras mode output:  [[1. 3.] [2. 4.]]
    - Lite mode output:  [[1. 2.] [3. 4.]]

```python
import tensorflow as tf
import numpy as np
x1 = tf.constant([1., 2., 3., 4.], shape=[2, 2, 1])

class Model(tf.keras.Model):

  def __init__(self):
    super(Model, self).__init__()

  @tf.function(input_signature=[tf.TensorSpec(x1.shape, x1.dtype)])
  def call(self, x):
    unpack_op = tf.raw_ops.Unpack(value=x,num=2,axis=0)
    return tf.concat(unpack_op, -1)
m = Model()
m(x1)
print('Keras mode output: ', m(x1).numpy())

converter = tf.lite.TFLiteConverter.from_keras_model(m)
tflite_model = converter.convert()

def _evaluateTFLiteModel(tflite_model, input_data):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for i in range(len(input_data)):
        interpreter.set_tensor(input_details[i]['index'], input_data[i])
    interpreter.invoke()
    output_data = [interpreter.get_tensor(output_details[i]['index'])
                   for i in range(len(output_details))]
    return output_data
print('Lite mode output: ', _evaluateTFLiteModel(tflite_model,[x1])[0])
tf.lite.experimental.Analyzer.analyze(model_content=tflite_model) #Output IR
```

#### Case 3: Model conversion fails.
-   Keras mode output: [[16. 19.]]
-   Lite mode output:  `RuntimeError: tensorflow/lite/kernels/fully_connected.cc:360 NumElements(bias) != SizeOfDimension(filter, 0) (1 != 2)Node number 0 (FULLY_CONNECTED) failed to prepare.Failed to apply the default TensorFlow Lite delegate indexed at 0`.
-   During conversion, the model fails due to the following check at `tensorflow/lite/kernels/fully_connected.cc:360`
-  `if (bias) {
    TF_LITE_ENSURE_EQ(context, NumElements(bias), SizeOfDimension(filter, 0));
  }`


```python
import tensorflow as tf
import numpy as np

x1 = tf.constant([1., 2.], shape=[1, 2])

class Model(tf.keras.Model):
  def __init__(self):
    super(Model, self).__init__()
    self.w = tf.Variable([[3., 4.], [5., 6.]])
    self.b = tf.Variable([3.])
  @tf.function(input_signature=[tf.TensorSpec(x1.shape, x1.dtype)])
  def call(self, x):
    return tf.matmul(x, self.w) + self.b


m = Model()
print('Keras mode output: ', m(x1).numpy())

converter = tf.lite.TFLiteConverter.from_keras_model(m)
tflite_model = converter.convert()
def _evaluateTFLiteModel(tflite_model, input_data):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for i in range(len(input_data)):
        interpreter.set_tensor(input_details[i]['index'], input_data[i])

    interpreter.invoke()

    output_data = [interpreter.get_tensor(output_details[i]['index'])
                   for i in range(len(output_details))]
    return output_data

print('Lite mode output: ', _evaluateTFLiteModel(tflite_model,[x1])[0])
```