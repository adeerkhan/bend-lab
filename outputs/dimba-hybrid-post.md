# Dimba hybrid — post draft

## Source
Own results: `results/hit-v4/{vit,jit,hit}/run.json`, `results/dimba/hybrid/run.json`,
`results/dimba/ablation-no-mamba/run.json`, and the one-page figure
`results/dimba/one_page.png`. No paper claim is reused as our result.
Source link for the card: https://github.com/adeerkhan/bend-lab/tree/main/results/dimba

## Audience
Curious engineers who follow diffusion and model architecture, not specialists.

## Voice
Plain and direct.

## User's thought (verbatim)
"write a Twitter Post, regenerate the Plots, one one single PNG of legibility, capacity, training loss plots, and then training data size, etc so that we can post them"

## Chosen lens
Cost / trade-off: the hybrid wins on our task, but the win is small and costs 5x the time; removing its Mamba blocks erases the win. This comes from the measured ablation, not a claim from the Dimba paper.

## Posts

1. Four small diffusion models, same flowers, same 1,200 steps. The plain transformer scored 0.514. The best hybrid scored 0.622. The difference is not the parameter count.

2. Here is the catch: the hybrid took 1,705 seconds; the others took about 500. Remove its Mamba blocks and the score drops back to exactly the plain model's. One recurrent layer did all the work.

3. The full comparison, one image and every number: https://github.com/adeerkhan/bend-lab/tree/main/results/dimba 🌐 These are small CPU reference models, not a reproduction of the paper's image quality.
