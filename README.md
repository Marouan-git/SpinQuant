# SpinQuant \- Mixed-Precision Analysis Fork

This repository is a fork of the original \[SpinQuant\](https://github.com/mit-han-lab/spinquant) implementation. While it retains the core functionality for learning optimal rotation matrices for LLM quantization, this version has been extended to support advanced analysis and optimization of **mixed-precision quantization**.

The primary additions include a suite of tools to:  
1\.  Analyze the sensitivity of different model components (layers, modules, or blocks) using various metrics.  
2\.  Generate optimized mixed-precision configurations based on a sensitivity metric and a computational budget.  
3\.  Evaluate the performance (Perplexity and Zero-Shot Accuracy) of these mixed-precision configurations.

## Setup

The setup is identical to the original SpinQuant repository. Please refer to the original [README](./README_SpinQuant.md) for instructions on installation and data preparation.

**Key Requirement**: Before running any scripts, ensure you are logged into your Hugging Face account via the CLI if you are using a gated model like Llama:  
```bash  
huggingface-cli login
```
---

## **Mixed-Precision Analysis and Optimization Workflow**

The workflow is designed to first analyze model sensitivity, then generate an optimal configuration, and finally evaluate it.

### **1\. Analyze Model Sensitivity (PPL Degradation or Top-k Stability)**

The first step is to run an analysis to determine how sensitive each module is to quantization. This fork provides two primary methods for this: PPL Degradation and Top-k Stability.

#### **a. Perplexity (PPL) Degradation Analysis**

**What it does**: This script iterates through every linear module in the model. For each module, it quantizes *only that module* to 4-bits and runs a full perplexity evaluation. The resulting PPL score is a direct measure of that module's sensitivity. The analysis is time-consuming but provides a very direct performance metric.

**Command**:

```
bash scripts/run_ppl_module_analysis.sh
```
* **Output**: The script will generate a JSON file in the ppl\_analysis/ directory (e.g., llama2-7b\_ppl\_per\_module.json), containing the perplexity score for each module.

#### **b. Top-k Tokens Stability Analysis**

**What it does**: This script also iterates through each module. For each, it compares the top-k predicted tokens (logits) of the baseline model (W4A16) against the model with that single module quantized. It calculates a "hybrid score" based on rank instability (Jaccard distance) and confidence shift (JSD). Modules that cause a larger change in the top-k predictions are considered more sensitive.

**Command**:

```
bash scripts/run_topk_module_analysis.sh
```
* **Output**: This will produce a JSON file in the topk\_analysis/ directory (e.g., llama2-7b\_topk20\_per\_module.json), containing the hybrid stability score for each module.

---

### **2\. Optimize Mixed-Precision Configuration**

**What it does**: After generating a sensitivity file (from step 1, or from another tool like LMAnalyser), this Python script uses the "Bang for the Buck" greedy algorithm to solve the Multiple-Choice Knapsack Problem. It takes the sensitivity scores, calculates different BOPs budgets, and generates the most optimal mixed-precision configuration for each budget.

**Command**:

```
python optimize_quant_config_multi_greedy.py  
    --model_path meta-llama/Llama-2-7b-hf  
    --sensitivity_metric ppl 
    --granularity module
    --bops_file_path ./bops_results/bops_per_module_llama2-7b.json
    --sensitivity_file_path ./ppl_analysis llama2-7b_ppl_per_module.json 
    --output_dir ./mixed_precision_configs_ppl
```
* \--sensitivity\_metric: The name of the metric used (e.g., ppl, topk10, fgmp\_block32). This is for naming the output files.  
* \--granularity: The level at which to apply precision (module, layer, or block).  
* \--bops\_file\_path: Path to the pre-computed BOPs for each component.  
* \--sensitivity\_file\_path: Path to the JSON file generated in step 1\.  
* \--output\_dir: Directory where the resulting configuration files will be saved.  
* **Output**: A set of JSON configuration files will be saved in the specified output directory, one for each budget point (e.g., best\_config\_0.25.json).

---

### **3\. Evaluate Mixed-Precision Configurations**

Once you have generated the optimized configurations, you can evaluate their performance.

#### **a. Perplexity Evaluation for a Mixed-Precision Config**

**What it does**: This script takes a mixed-precision configuration JSON file and runs a full perplexity evaluation on the model quantized according to that configuration.

**Command**:

```
bash scripts/run_mixed_precision_ppl_eval.sh
```

* **Inside the script**, you must set the config\_path variable to point to the specific folder containing the configuration files you want to evaluate.  
* **Output**: The script will print the final perplexity score to the console and save the detailed results in a JSON file.

#### **b. Zero-Shot Accuracy Evaluation for a Mixed-Precision Config**

**What it does**: This script evaluates a given mixed-precision configuration on a suite of five zero-shot commonsense reasoning tasks (ARC-easy, ARC-challenge, PIQA, HellaSwag, Winogrande) using the lm-evaluation-harness.

**Command**:

```
bash scripts/run_mixed_precision_zero_shot_eval.sh
```

* Similar to the perplexity script, you need to edit the config\_path variable inside the .sh file to point to the desired configuration.  
* **Output**: The script will output the accuracy for each task and the mean accuracy. Results are saved to JSON files in the zero-shot\_results/ directory.

---

### **4\. Plotting and General Notes**

* The plotting/ directory contains various Python scripts (plot\_mixed\_precision\_results.py, plot\_sensitivity.py, etc.) to visualize the results from the analysis and evaluation steps.  
* **Important**: The shell scripts in the scripts/ directory may contain hardcoded paths (e.g., for models, datasets, or output files). You may need to edit these paths to match your local environment setup.