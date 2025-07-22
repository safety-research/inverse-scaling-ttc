# Inverse Scaling in Test-Time Compute

[![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/2507.14417) [![Demo](https://img.shields.io/badge/Demo-Live-green.svg)](https://)

## 📝 Abstract

> We construct evaluation tasks where extending the reasoning length of Large Reasoning Models (LRMs) deteriorates performance, exhibiting an inverse scaling relationship between test-time compute and accuracy. Our evaluation tasks span four categories: simple counting tasks with distractors, regression tasks with spurious features, deduction tasks with constraint tracking, and advanced AI risks. We identify five distinct failure modes when models reason for longer: 1) Claude models become increasingly distracted by irrelevant information; 2) OpenAI o-series models resist distractors but overfit to problem framings; 3) models shift from reasonable priors to spurious correlations; 4) all models show difficulties in maintaining focus on complex deductive tasks; and 5) extended reasoning may amplify concerning behaviors, with Claude Sonnet 4 showing increased expressions of self-preservation. These findings suggest that while test-time compute scaling remains promising for improving model capabilities, it may inadvertently reinforce problematic reasoning patterns. Our results demonstrate the importance of evaluating models across diverse reasoning lengths to identify and address these failure modes in LRMs.

## 🚀 Quick Start

```bash
# Clone and set up
git clone --recurse-submodules https://github.com/safety-research/inverse-scaling-ttc.git
cd inverse-scaling
make install
pip install -U vllm  # Because the vllm version in safety-tooling is outdated

# Configure API keys
cp .env.example .env  # Add your ANTHROPIC_API_KEY_NORMAL_BATCH, OPENAI_API_KEY, HF_TOKEN

# Download datasets and validate
make download-data
make validate
```



## 📚 Task Categories

- **Main Tasks**: synthetic_misleading_math, synthetic_misleading_python, student_lifestyle_regression, mathematical_paradoxes, zebra_puzzles
- **AI Risk Tasks**: survival instinct, corrigibility, self-awareness, power/wealth-seeking, etc.
- **Inverse Scaling Prize**: hindsight-neglect, memo-trap, modus-tollens, pattern-matching-suppression, etc.
- **Standard Benchmarks**: GSM8K, ASDIV, MultiArith, etc.


## 🧪 Running Experiments

Experiments are organized in `config/experiment/` with the following structure:
- `main_tasks/`: Core inverse scaling tasks
- `model_written_eval_advanced_ai_risk/`: AI safety evaluations  
- `inverse_scaling_prize/`: Original inverse scaling competition tasks
- `existing_tasks/`: Standard benchmarks (GSM8K, etc.)

### Examples

```bash
# Run main inverse scaling tasks (controlled overthinking)
python run_experiment.py experiment=main_tasks/claude4o
python run_experiment.py experiment=main_tasks/o3
python run_experiment.py experiment=main_tasks/deepseek_r1_0528_awq

# Run AI risk evaluations
python run_experiment.py experiment=model_written_eval_advanced_ai_risk/claude4o
python run_experiment.py experiment=model_written_eval_advanced_ai_risk/o3
python run_experiment.py experiment=model_written_eval_advanced_ai_risk/deepseek_r1_0528_awq

# Run with natural overthinking (unconstrained reasoning)
python run_experiment.py experiment=main_tasks/claude4o_natural_overthinking

# VLLM open-source models
python run_experiment.py experiment=main_tasks/qwen3_32b
python run_experiment.py experiment=main_tasks/qwq_32b

# Override specific parameters
python run_experiment.py experiment=main_tasks/claude4o \
  evaluation.batch_size=50 \ # Good for local debugging
  wandb.enabled=false

# Custom reasoning budgets
python run_experiment.py experiment=main_tasks/qwen3_32b \
  evaluation.reasoning_budgets=[0,1024,4096,8192]
```


## 📁 Project Structure

```
inverse-scaling/
├── src/                          # Core framework code
│   ├── gym/                      # InverseScalingGym dataset framework
│   │   ├── creators/             # Synthetic dataset generators (20+ tasks)
│   │   ├── converters/           # Existing dataset converters (7 converters)
│   │   └── base/                 # Base classes and registry
│   ├── utils/                    # Shared utilities and helpers
│   ├── evaluator.py             # Core evaluation orchestration
│   ├── model_interface.py       # Base model interface with caching
│   ├── batch_model_interface.py # Batch processing interface
│   ├── vllm_model_interface.py  # VLLM interface for open-source models
│   ├── task_loader.py           # Task loading and preprocessing
│   └── results_manager.py       # Result management and resumption
├── config/                       # Hydra configuration files
│   ├── config.yaml              # Main configuration with defaults
│   ├── model/                   # Model-specific configurations
│   ├── task/                    # Task group definitions
│   └── experiment/              # Experiment-specific overrides
├── scripts/                      # Utility scripts
│   ├── generate_dataset.py      # InverseScalingGym CLI interface
│   ├── analyze_results_multimodel.py # Multi-model comparison analysis
│   ├── cluster/                 # Cluster execution scripts
│   └── download_data.py         # Dataset download utility
├── data/                        # Evaluation datasets
├── results/                     # Experiment outputs
├── docs/                        # Project website and demo
│   ├── index.html              # Interactive demo
│   ├── script.js               # Demo functionality
│   └── demo_data.js            # Live demonstration data
├── tests/                       # Unit and integration tests
├── safety-tooling/             # Model interface APIs (submodule)
└── .logs/                      # Automated log file storage
```


## 🧪 Development

```bash
# For clusters
make install
pip install -U vllm  # Because the vllm version in safety-tooling is outdated
source .venv/bin/activate
./scripts/cluster/run_all_experiments.sh -s session_name experiment1 experiment2
```

## 📄 Citation

If you use this framework in your research, please cite:

```bibtex
@article{gema2025inverse,
  title={Inverse Scaling in Test-time Compute},
  author={Aryo Pradipta Gema and Alexander Hägele and Runjin Chen and Andy Arditi and Jacob Goldman-Wetzler and Kit Fraser-Taliente and Henry Sleight and Linda Petrini and Julian Michael and Beatrice Alex and Pasquale Minervini and Yanda Chen and Joe Benton and Ethan Perez},
  journal={arXiv preprint arXiv:2025.14417},
  year={2025}
}
```


## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


---

**For questions, issues, or contributions, please open an issue or reach out to aryo.gema@ed.ac.uk**