# See How It Thinks: Mixed Palletizing with Explainable Visual Reasoning

> **Authors:** [Yuri Rocha](https://github.com/doosan-robotics/explainable-palletizer) • Yujeong Jeong • Minsoo Song • Kyungchan Son
> **Organization:** Doosan Robotics (Zenith Team)

## Overview

| **Model** | **Workload** | **Use Case** |
|-----------|--------------|--------------|
| Cosmos Reason2 8B | Post-Training + Inference | Explainable robotic palletizing with visual reasoning |

Robots in warehouse automation typically handle boxes blindly — following fixed rules regardless of contents, condition, or fragility. This recipe demonstrates an intelligent palletizing system that uses Cosmos Reason2 8B, fine-tuned with LoRA on synthetic Isaac Sim data, to visually infer box properties and generate auditable chain-of-thought placement decisions — without barcodes or labels.

- **GitHub Repository:** [doosan-robotics/explainable-palletizer](https://github.com/doosan-robotics/explainable-palletizer)
- **Demo Video:** [YouTube](https://youtu.be/kV6m2Ab6zng)

> **Recommended Structure:** This end-to-end recipe covers the full pipeline from synthetic data generation through fine-tuning to edge deployment. Content is organized as:
>
> 1. **Problem & Use Case** - Define the real-world application and value proposition
> 2. **Model Selection** - Why Cosmos Reason2 fits (chain-of-thought visual reasoning out-of-the-box)
> 3. **Data Pipeline** - Synthetic data generation with Isaac Sim
> 4. **Post-Training** - LoRA fine-tuning on synthetic palletizing data
> 5. **Inference Pipeline** - vLLM serving + containerized control loop
> 6. **Edge Deployment** - Jetson Thor deployment
> 7. **Results & Impact** - Accuracy benchmarks and real-world validation

## Key Features

- **Explainable decisions**: Chain-of-thought reasoning traces make every placement decision auditable
- **No barcode dependency**: Infers box contents, weight class, and fragility from camera image alone
- **Damage detection**: Routes unsafe boxes to human inspection before palletizing
- **LoRA efficiency**: Fine-tuning on synthetic data achieves 65% action accuracy (+20% over zero-shot)

## How It Works

1. **Synthetic data generation**: Isaac Sim renders labeled box scenarios with known ground-truth placement parameters
2. **LoRA fine-tuning**: Cosmos Reason2 8B is fine-tuned on synthetic VQA pairs covering box classification, damage detection, and placement reasoning
3. **Containerized inference**: Four services manage a continuous perception-to-execution loop; vLLM serves the fine-tuned model
4. **Motion execution**: cuRobo translates model placement outputs into GPU-accelerated motion plans
5. **Edge deployment**: The full stack runs on Jetson Thor for on-robot inference

## Dataset and Setup

### Input Data

Images captured from robot-mounted cameras showing mixed pallets. For training: synthetic renders from Isaac Sim with structured VQA annotations.

### Data Structure

```
data/
  synthetic/
    images/          # Isaac Sim renders
    annotations/     # VQA pairs: question, chain-of-thought, answer
  real/
    validation/      # Real warehouse images for evaluation
```

## Pipeline Components

### Component 1: Synthetic Data Generation (Isaac Sim)

[Detailed explanation with Isaac Sim scene configuration and export scripts]

```python
# Example: export synthetic VQA pair from Isaac Sim scene
```

### Component 2: LoRA Fine-Tuning

[Detailed explanation of LoRA configuration and training loop]

```bash
# Example: launch fine-tuning job
```

### Component 3: vLLM Inference Service

[Detailed explanation of vLLM serving configuration for Cosmos Reason2]

```bash
# Example: launch vLLM server with fine-tuned adapter
```

### Component 4: Jetson Thor Edge Deployment

[Containerized deployment instructions for Jetson Thor]

## Results

- **Format compliance**: 100% (structured output parsing)
- **Action accuracy**: 65% (+20% over zero-shot baseline) via five-fold cross-validation
- **Damage detection**: [metric TBD]

## Conclusion

By fine-tuning Cosmos Reason2 8B with LoRA on synthetic Isaac Sim data, this recipe demonstrates that warehouse robots can move from blind rule-following to reasoned, explainable placement decisions using only camera input.

**Potential Applications:**

- Mixed-SKU warehouse palletizing without barcode infrastructure
- Fragile goods handling with automatic damage routing
- Auditable robotic decision logs for compliance and QA

**Next Steps:**

- Extend synthetic dataset diversity (more box types, lighting conditions)
- Evaluate on multi-robot coordination scenarios
- Optimize Jetson Thor inference latency

## Resources

- **[explainable-palletizer repo](https://github.com/doosan-robotics/explainable-palletizer)** - Full source code, training scripts, and deployment configs
- **[Demo Video](https://youtu.be/kV6m2Ab6zng)** - Live system demonstration
- **[Cosmos Reason2](https://huggingface.co/collections/nvidia/cosmos-reason1-67c9bc2e4e5e3cf8b22e0167)** - Base model on HuggingFace
- **[Isaac Sim](https://developer.nvidia.com/isaac/sim)** - Synthetic data generation platform
- **[cuRobo](https://curobo.org/)** - GPU-accelerated motion planning
