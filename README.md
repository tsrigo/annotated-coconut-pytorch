<img src="./coconut.png" width="400px"></img>

## 🥥 Coconut (wip)

Implementation of Coconut, proposed by the paper <a href="https://arxiv.org/abs/2412.06769">Training Large Language Models to Reason in a Continuous Latent Space</a> out of FAIR, in Pytorch

Architecture wise, the closest work to the one proposed here would be <a href="https://github.com/lucidrains/recurrent-memory-transformer-pytorch">RMT</a>, where the memory tokens there could serve as the continuous latent tokens. Both directions are worth exploring

## Install

```bash
$ pip install coconut-pytorch
```

## Usage

```python
import torch
from coconut_pytorch import Coconut

model = Coconut(
    num_reasoning_steps = 3,
    transformer = dict(
        num_tokens = 256,
        dim = 512,
        depth = 2
    )
)

prompt = torch.randint(0, 256, (1, 1024))
answer = torch.randint(0, 256, (1, 64))

prompt_logits, reasoning_tokens, answer_logits = model(prompt, answer)

```

## Citation

```bibtex
@inproceedings{Hao2024TrainingLL,
    title   = {Training Large Language Models to Reason in a Continuous Latent Space},
    author  = {Shibo Hao and Sainbayar Sukhbaatar and DiJia Su and Xian Li and Zhiting Hu and Jason Weston and Yuandong Tian},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:274610816}
}
```
