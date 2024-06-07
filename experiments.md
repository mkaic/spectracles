### Session 1 Experiments
1. 16.3% acc. 12k params. 32 x 4, 40 epochs. 256 batch size.
2. 28.5% acc. 12k params. 32 x 4, 120 epochs. 256 batch size.
3. 31.3% acc. 40k params. 64 x 4, 150 epochs. 256 batch size.
4. 2.1% acc. 21k params. 32 x 8, 100 epochs. 256 batch size. Took 80 epochs just to get out of super-high-loss, maybe due to lack of normalization?
5. 10.1% acc. 21k params. 30 epochs. Same as (4) but **normalizing** along dims (1)
6. 4.3% acc. 21k params. 20 epochs. Same as (4) but **normalizing** along dims (2,3)
7. 29.1% acc. 21k params. 50 epochs. Same as (4) but **standardizing** along dims (1,2,3)
8. 16.0% acc. 21k params. 50 epochs. Same as (4) but **standardizing** along dims (1,)
9. 13.1% acc. 12k params. 32 x 4, 30 epochs. **Standardizing** along dims (1,)
10. 17.0% acc. 12k params. 50 epochs. Same as (9) but without residual connections.
11. 25.5% acc. 12k params. 50 epochs. Same as (9) but **standardizing** along dims (1,2,3)

### Session 1 Findings
Standardizing every image in every block helps. Residual connections don't seem to hurt or help. Deeper vs wider layers is unclear. Standardizing works better than Normalizing.

### Session 2 Experiments
12. 31% acc. 17k params. 120 epochs. Same as (11) but with an extra "linear layer" in every block.

Further experiments are happening at [this WandB workspace](https://wandb.ai/mkaichristensen/spectracles).