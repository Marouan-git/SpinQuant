import json
import argparse
from itertools import product
import random

def load_and_prepare_data(args):
    """Loads data and prepares a list of upgrades sorted by efficiency."""
    
    # --- Load Data based on Granularity ---
    if args.granularity == 'layer':
        with open(args.bops_layer, 'r') as f: bops_data = json.load(f)
        with open(args.ppl_layer_4bit, 'r') as f: ppl_data_4 = json.load(f)
        with open(args.ppl_layer_8bit, 'r') as f: ppl_data_8 = json.load(f)
        if args.metric == 'ratio':
            with open(args.ratio_layer, 'r') as f: ratio_data = {int(k): v for k, v in json.load(f).items()}
        elif args.metric == 'fgmp':
            with open(args.fgmp_layer_4_8, 'r') as f: fgmp_data_4_8 = {int(k): v for k, v in json.load(f).items()}
            with open(args.fgmp_layer_8_16, 'r') as f: fgmp_data_8_16 = {int(k): v for k, v in json.load(f).items()}
        elif args.metric == 'fisher' or args.metric == 'fisher_lse':
            with open(args.fisher_layer, 'r') as f: fisher_data = {int(k): v for k, v in json.load(f).items()}
        elif args.metric == 'topk':
            with open(args.topk_layer, 'r') as f:
                topk_data_list = json.load(f)
                topk_data = {item['layer_id']: item['hybrid_score'] for item in topk_data_list}
        
        # Get number of keys/layers
        num_layers = len(bops_data)
        print(f"Number of layers detected: {num_layers}")
        quantizable_units = list(range(num_layers))
        base_gmacs = {i: bops_data[str(i)] / (4*4) for i in quantizable_units}

    elif args.granularity == 'module':
        with open(args.bops_module, 'r') as f: bops_data = json.load(f)
        with open(args.ppl_module_4bit, 'r') as f: ppl_data_4 = json.load(f)
        with open(args.ppl_module_8bit, 'r') as f: ppl_data_8 = json.load(f)
        if args.metric == 'ratio':
            with open(args.ratio_module, 'r') as f: ratio_data_nested = json.load(f)
            # Flatten the nested ratio data
            ratio_data = {}
            for module_type, layer_ratios in ratio_data_nested.items():
                for layer_idx, ratio in layer_ratios.items():
                    # Construct the full module name
                    module_name = f"model.layers.{layer_idx}.{module_type}"
                    ratio_data[module_name] = ratio
        elif args.metric == 'topk':
            with open(args.topk_module, 'r') as f:
                topk_data_list = json.load(f)
                # Parse the list of objects into a dictionary of {module_name: hybrid_score}
                topk_data = {item['module_name']: item['hybrid_score'] for item in topk_data_list}
        elif args.metric == 'fgmp':
            with open(args.fgmp_module_4_8, 'r') as f: fgmp_data_4_8_nested = json.load(f)
            fgmp_data_4_8 = {f"model.layers.{l}.{t}": r for t, d in fgmp_data_4_8_nested.items() for l, r in d.items()}
            with open(args.fgmp_module_8_16, 'r') as f: fgmp_data_8_16_nested = json.load(f)
            fgmp_data_8_16 = {f"model.layers.{l}.{t}": r for t, d in fgmp_data_8_16_nested.items() for l, r in d.items()}
        elif args.metric == 'fisher' or args.metric == 'fisher_lse':
            with open(args.fisher_module, 'r') as f: fisher_data_nested = json.load(f)
            fisher_data = {f"model.layers.{l}.{t}": r for t, d in fisher_data_nested.items() for l, r in d.items()}
        

        # Define all possible module names as the quantizable units
        quantizable_units = [item['module_name'] for item in ppl_data_4 if 'baseline' not in item['module_name']]
        base_gmacs = {unit: bops_data[unit.split('.')[2]][unit.split('.', 3)[-1]] / (4*4) for unit in quantizable_units}

    elif args.granularity == 'block':
        # For block-level, we define units as (module_name, block_index)
        with open(args.bops_block, 'r') as f: bops_data = json.load(f)
        with open(args.ppl_module_4bit, 'r') as f: ppl_data_4 = json.load(f)
        with open(args.ppl_module_8bit, 'r') as f: ppl_data_8 = json.load(f)
        
        # The 'fgmp' metric is required for block granularity in this implementation
        with open(args.fgmp_block_4_8, 'r') as f: fgmp_data_4_8 = json.load(f)
        with open(args.fgmp_block_8_16, 'r') as f: fgmp_data_8_16 = json.load(f)

        quantizable_units = []
        for module_name, block_values in bops_data.items():
            for i in range(len(block_values)):
                quantizable_units.append((module_name, i))
        
        base_gmacs = {(mod, i): bops_data[mod][i] / (4*4) for mod, i in quantizable_units}

    # --- Prepare Profiles ---
    fixed_w_bits = 4
    bops_profile = {unit: {p: base_gmacs[unit] * fixed_w_bits * p for p in [4, 8, 16]} for unit in quantizable_units}
    
    # PPL data is loaded for all modes for final estimation
    id_key = 'module_name' if args.granularity in ['module', 'block'] else 'layer_id'
    baseline_ppl = min(item['perplexity'] for item in ppl_data_4)
    ppl_degradation_4bit_raw = {item[id_key]: item['perplexity'] - baseline_ppl for item in ppl_data_4}
    ppl_degradation_8bit_raw = {item[id_key]: item['perplexity'] - baseline_ppl for item in ppl_data_8}

    # Distribute module PPL degradation across blocks if needed (heuristic)
    if args.granularity == 'block':
        ppl_degradation_4bit, ppl_degradation_8bit = {}, {}
        num_blocks_per_module = {mod: len(vals) for mod, vals in bops_data.items()}
        for module_name, num_blocks in num_blocks_per_module.items():
            for i in range(num_blocks):
                ppl_degradation_4bit[(module_name, i)] = ppl_degradation_4bit_raw.get(module_name, 0) / num_blocks
                ppl_degradation_8bit[(module_name, i)] = ppl_degradation_8bit_raw.get(module_name, 0) / num_blocks
    else:
        ppl_degradation_4bit, ppl_degradation_8bit = ppl_degradation_4bit_raw, ppl_degradation_8bit_raw
    

    # --- Build and Sort Upgrades ---
    upgrades = []
    for unit_id in quantizable_units:
        if args.metric == 'ppl':
            improvement_4_to_8 = ppl_degradation_4bit.get(unit_id, 0) - ppl_degradation_8bit.get(unit_id, 0)
            improvement_8_to_16 = ppl_degradation_8bit.get(unit_id, 0)
        elif args.metric == 'ratio' or args.metric == 'topk':
            data_source = ratio_data if args.metric == 'ratio' else topk_data
            total_value = data_source.get(unit_id, 0)
            improvement_4_to_8 = total_value * 0.8
            improvement_8_to_16 = total_value * 0.2
        elif args.metric == 'fisher' or args.metric == 'fisher_lse':
            improvement_4_to_8 = fisher_data.get(unit_id, 0) * 0.8
            improvement_8_to_16 = fisher_data.get(unit_id, 0) * 0.2
        elif args.metric == 'fgmp':
            if args.granularity == 'block':
                module_name, block_idx = unit_id
                improvement_4_to_8 = fgmp_data_4_8.get(module_name, [])[block_idx]
                improvement_8_to_16 = fgmp_data_8_16.get(module_name, [])[block_idx]
            else: # Layer or Module
                improvement_4_to_8 = fgmp_data_4_8.get(unit_id, 0)
                improvement_8_to_16 = fgmp_data_8_16.get(unit_id, 0)
        
        cost_4_to_8 = bops_profile[unit_id][8] - bops_profile[unit_id][4]
        if cost_4_to_8 > 0:
            upgrades.append({'id': unit_id, 'from': 4, 'to': 8, 'bang_for_buck': improvement_4_to_8 / cost_4_to_8, 'cost': cost_4_to_8})

        cost_8_to_16 = bops_profile[unit_id][16] - bops_profile[unit_id][8]
        if cost_8_to_16 > 0:
            upgrades.append({'id': unit_id, 'from': 8, 'to': 16, 'bang_for_buck': improvement_8_to_16 / cost_8_to_16, 'cost': cost_8_to_16})

    sorted_upgrades = sorted(upgrades, key=lambda x: x['bang_for_buck'], reverse=True)
    return quantizable_units, bops_profile, sorted_upgrades, baseline_ppl, ppl_degradation_4bit, ppl_degradation_8bit


def find_best_config_greedy(bops_budget, quantizable_units, bops_profile, sorted_upgrades):
    """Finds the optimal quantization config using the greedy algorithm."""
    current_config = {unit: 4 for unit in quantizable_units}
    current_bops = sum(bops_profile[unit][4] for unit in quantizable_units)

    for upgrade in sorted_upgrades:
        unit_id = upgrade['id']
        if current_config[unit_id] == upgrade['from'] and current_bops + upgrade['cost'] <= bops_budget:
            current_config[unit_id] = upgrade['to']
            current_bops += upgrade['cost']
    return current_config, current_bops

def find_random_config_constructive(bops_budget, quantizable_units, bops_profile, all_upgrades):
    """Finds a random config by shuffling upgrades and applying them until budget is met."""
    # Start with the cheapest configuration
    current_config = {unit: 4 for unit in quantizable_units}
    current_bops = sum(bops_profile[unit][4] for unit in quantizable_units)
    
    # Randomly shuffle the list of all possible upgrades
    random.shuffle(all_upgrades)
    
    # Apply upgrades in random order until the budget is met
    for upgrade in all_upgrades:
        unit_id = upgrade['id']
        if current_config[unit_id] == upgrade['from'] and current_bops + upgrade['cost'] <= bops_budget:
            current_config[unit_id] = upgrade['to']
            current_bops += upgrade['cost']
    return current_config, current_bops


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find best quant config with a greedy algorithm.')
    parser.add_argument('--model_name', type=str, default="Llama-2-7b", help='Name of the model to optimize.')
    parser.add_argument('--mode', type=str, default='greedy', choices=['greedy', 'random'], help='The method to use for finding the configuration.')
    parser.add_argument('--granularity', type=str, required=True, choices=['layer', 'module', 'block'], help='Optimization granularity.')
    parser.add_argument('--metric', type=str, default='ppl', choices=['ppl', 'ratio', 'fgmp', 'fisher', 'fisher_lse', 'topk'], help='Sensitivity metric.')
    # Layer-wise files
    parser.add_argument('--ppl-layer-4bit', type=str, help='Layer-wise PPL sensitivity file for 4-bit.')
    parser.add_argument('--ppl-layer-8bit', type=str, help='Layer-wise PPL sensitivity file for 8-bit.')
    parser.add_argument('--ratio-layer', type=str, help='Layer-wise ratio file.')
    parser.add_argument('--fgmp-layer-4-8', type=str, help='Layer-wise FGMP 4 to 8 file.')
    parser.add_argument('--fgmp-layer-8-16', type=str, help='Layer-wise FGMP 8 to 16 file.')
    parser.add_argument('--fisher-layer', type=str, help='Layer-wise Fisher Information file.')
    parser.add_argument('--topk-layer', type=str, help='Layer-wise TopK hybrid score file.')
    parser.add_argument('--bops-layer', type=str, help='Layer-wise BOPs file.')
    # Module-wise files
    parser.add_argument('--ppl-module-4bit', type=str, help='Module-wise PPL sensitivity file for 4-bit.')
    parser.add_argument('--ppl-module-8bit', type=str, help='Module-wise PPL sensitivity file for 8-bit.')
    parser.add_argument('--ratio-module', type=str, help='Module-wise ratio file.')
    parser.add_argument('--fgmp-module-4-8', type=str, help='Module-wise FGMP 4 to 8 file.')
    parser.add_argument('--fgmp-module-8-16', type=str, help='Module-wise FGMP 8 to 16 file.')
    parser.add_argument('--fisher-module', type=str, help='Module-wise Fisher Information file.')
    parser.add_argument('--topk-module', type=str, help='Module-wise TopK hybrid score file.')
    parser.add_argument('--bops-module', type=str, help='Module-wise BOPs file.')
    # Block-wise files
    parser.add_argument('--fgmp-block-4-8', type=str, help='Block-wise FGMP 4-to-8 file.')
    parser.add_argument('--fgmp-block-8-16', type=str, help='Block-wise FGMP 8-to-16 file.')
    parser.add_argument('--bops-block', type=str, help='Block-wise BOPs file.')
    
    parser.add_argument('--budget_multiplier', type=float, default=0.5, help='Budget relative to cost range.')
    args = parser.parse_args()
    
    # --- Load data and prepare upgrades based on args ---
    quantizable_units, bops_profile, sorted_upgrades, baseline_ppl, ppl_degradation_4bit, ppl_degradation_8bit = load_and_prepare_data(args)
    
    # --- Calculate Budget ---
    full_low_prec_cost = sum(bops_profile[unit][4] for unit in quantizable_units)
    middle_prec_cost = sum(bops_profile[unit][8] for unit in quantizable_units)
    full_high_prec_cost = sum(bops_profile[unit][16] for unit in quantizable_units)

    m = args.budget_multiplier

    if m <= 0.5:
        # Interpolate between low and middle precision costs
        scale = 2 * m
        BOPs_BUDGET = full_low_prec_cost + scale * (middle_prec_cost - full_low_prec_cost)
    else:
        # Interpolate between middle and high precision costs
        scale = 2 * (m - 0.5)
        BOPs_BUDGET = middle_prec_cost + scale * (full_high_prec_cost - middle_prec_cost)
    
    print(f"Granularity: {args.granularity} | Metric: {args.metric}")
    print(f"Target BOPs Budget: {BOPs_BUDGET / 1e3:.2f} T-BOPs") # Assuming G-BOPs input
    print("-" * 40)
    
    if args.mode == 'random':
        # --- Find a random configuration under the budget ---
        best_config, final_bops = find_random_config_constructive(BOPs_BUDGET, quantizable_units, bops_profile, sorted_upgrades)
    else:
        # --- Run Greedy Algorithm ---
        best_config, final_bops = find_best_config_greedy(BOPs_BUDGET, quantizable_units, bops_profile, sorted_upgrades)
    
    # --- Estimate Final PPL and Print Results ---
    final_ppl = baseline_ppl
    for unit_id, bits in best_config.items():
        if bits == 4:
            final_ppl += ppl_degradation_4bit.get(unit_id, 0)
        elif bits == 8:
            final_ppl += ppl_degradation_8bit.get(unit_id, 0)

    print("Greedy Optimization Finished!")
    print(f"Best Estimated PPL: {final_ppl:.4f}")
    print(f"Achieved BOPs: {final_bops / 1e3:.2f} / {BOPs_BUDGET / 1e3:.2f} T-BOPs")
    print("Best Quantization Configuration:")
   
    

    # Convert integer keys to strings for JSON compatibility
    if args.granularity == 'block':
        best_config_str_keys = {f"{k[0]}.block_{k[1]}": v for k, v in best_config.items()}
    else:
        best_config_str_keys = {str(k): v for k, v in best_config.items()}
    
    print(json.dumps(best_config_str_keys, indent=4))

    with open(f'best_config_{args.model_name}_{args.granularity}_{args.metric}_{args.budget_multiplier}.json', 'w') as f:
        json.dump(best_config_str_keys, f, indent=4)
    print(f"Best configuration saved to 'best_config_{args.model_name}_{args.granularity}_{args.metric}_{args.budget_multiplier}.json'")
    
    # Calculate the percentage of layers for each bit-width and print it
    total_layers = len(best_config)
    bit_distribution = {4: 0, 8: 0, 16: 0}
    for bits in best_config.values():
        bit_distribution[bits] += 1

    # Print the percentage distribution
    for bits, count in bit_distribution.items():
        percentage = (count / total_layers * 100) if total_layers > 0 else 0
        print(f"Percentage of layers with {bits}-bit quantization: {percentage:.2f}%")
    
    # Export the percentages to a JSON file
    with open(f'bit_distribution_{args.model_name}_{args.granularity}_{args.metric}_{args.budget_multiplier}.json', 'w') as f:
        for bits, count in bit_distribution.items():
            bit_distribution[bits] = (count / total_layers * 100) if total_layers > 0 else 0
        json.dump(bit_distribution, f, indent=4)
    print(f"Bit distribution saved to 'bit_distribution_{args.model_name}_{args.granularity}_{args.metric}_{args.budget_multiplier}.json'")