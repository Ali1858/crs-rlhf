# Combined Preference-Based and Absolute Reward Signals for RLHF Fine-tuning
This repository contains the code and resources for my thesis on Reinforcement Learning from Human Feedback (RLHF) fine-tuning. The thesis was supervised by [Jun.-Prof. Dr. Dennis Riehle](https://www.uni-koblenz.de/de/informatik/iwvi/riehle/team/dennis-riehle) and [Anna Wolters, M.Sc.](https://www.uni-koblenz.de/de/informatik/iwvi/riehle/team/anna-wolters), and submitted to [The Institute of Information Systems](https://www.uni-koblenz.de/de/informatik/iwvi) at the University of Koblenz.

## Abstract

In the field of RLHF, the existing methods for fine-tuning center around a preference-based
reward model. While this approach is widely used, it lacks a global perspective on the quality of the individual samples. Since preference feedback is provided through pairwise comparison, it
can only offer relative information and does not account for the absolute quality of each
instance. This methodology, although practical, fails to measure how good or bad the examples
are, providing only an option to choose the best among the given and not
consider if all the given examples are of poor quality. To address these limitations, we trained the reward model directly using absolute feedback, which explicity learns from each response independently.

Motivated by the hypothesis that absolute rewards may offer a direct measure of response quality, which
could enable a more precise alignment of LLMs with human intention, this research
fills the gap by empirically investigating the impact of absolute reward signals on
RLHF fine-tuning for text generation.

This master's thesis explores the efficacy of incorporating absolute
reward signals into Reinforcement Learning from Human Feedback fine-tuning.
Furthermore, it study the trade-off between preference-based and absolute reward when combined. The investigation reveals that model trained with an absolute reward model
outperform those trained with a preference reward model. It found an inverse correlation
between the preference reward model's weight and performance when the reward scores of both
models are combined. The absolute reward model demonstrates better generalisability
over the preference reward model, highlighting its robust feedback mechanism
during fine-tuning. According to the experiments, the model fine-tuned with absolute and combined
reward signals achieved win rates of 66% and 63% respectively, outperforming models
fine-tuned solely with preference feedback.

## Data
This research mainly utilizes [Open Assistant](https://huggingface.co/datasets/OpenAssistant/oasst1) dataset for Supervised-Finetuning (SFT), Reward Modelling, and Reinforcement Learning (RL) fine-tuning.

## Reproduction
Make sure to choose the proper settings in the `config.yaml` file and install all requirements.
```bash
pip install -r requirements.txt
```

### Step 1: SFT
```bash
python sft_train.py --config_subset  sft --name_suffix _bs64_ep_1
```

### Step 2: Reward Modeling

For preference reward model training:
```bash
python rm_train.py --config_subset  rm --name_suffix _preference
```
Update the config and similarly train the Absolute reward model.

### Step 3: RL fine-tuning
```bash
python rl_train.py --config_subset  rl --name_suffix
```

Trained Lora adapters for [preference reward model](https://huggingface.co/alikhan0100u/Llama-2-7b-oasst-preference-reward-model-adapter) and [absolute reward model](https://huggingface.co/alikhan0100u/Llama-2-7b-oasst-abs-reward-model-adapter), along with all three RLHF (Abs_RLHF, Preference_RLHF, CRS_RLHF) models are available on [Huggingface](https://huggingface.co/alikhan0100u). Each RLHF model can be tested by generating response for 100 prompts from the [alpaca eval](https://huggingface.co/datasets/tatsu-lab/alpaca_eval) dataset using the script [rl_eval_pred.py](evaluations/rl_eval_pred.py) and then predicting reward score using the script [rl_eval_rewarding](evaluations/rl_eval_rewarding.py).

Result from automatic evaluation using GPT-4 can be found [here](alpaca_eval) and results from user-study can be found [here](evaluations/argilla_data). The results from our experiments are detailed further in the [documentation](docs)


## Acknowledgments and References
The training process of SFT and the reward model uses the Hugging Face's [Transformers](https://github.com/huggingface/transformers) library and follows the training procedure similar to the [Open Assistant](https://github.com/LAION-AI/Open-Assistant). For both models, our research substantially uses the code from Open Assitant and repurpose it for our specific use case. Whereas, RL fine-tuning utilizes the [TRL](https://github.com/huggingface/trl/tree/main) library and follows their training examples as a guide. Automated evaluation is done using [Alpaca eval](https://github.com/tatsu-lab/alpaca_eval/tree/main) library, whereas the portal for user-study was created using [Argilla](https://github.com/argilla-io/argilla).
