# hifi-gan
Implementation of the [HiFi-GAN](https://arxiv.org/abs/2010.05646) neural vocoder.

# Reproducing results
To reproduce trainig of the final model, follow the steps:

1. Specify the `GPUS` (gpu indices that will be used during training), `SAVE_DIR` (directory where all the logs & checkpoints will be stored), `DATA_DIR` (directory that will store the training data), `NOTEBOOK_DIR` (directory that contains your notebooks, for debugging purposes) in the `Makefile`. Set up `WANDB_API_KEY` variable in the `Dockerfile` to log the training process.

2. Build and run the `Docker` container
```bash
make build && make run
```

3. Run the pipeline described in the `configs/hifigan.json` configuration file.
```bash
python3 train.py --config configs/hifigan.json
```

# Running tests
1. In order to run an inference on pre-trained model, you should first download its weights by running
```bash
python3 install_weights.py
```

2. Copy the config file into the same directory as the model weights
```bash
cp configs/hifigan.json saved/models/final/
```

3. Run
```bash
python3 test.py -r saved/models/final/weight.pth -t <PATH_TO_AUDIO_DIR> -o <OUTPUT_DIR>
```

4. Listen to generated audios, that are now stored in the `<OUTPUT_DIR>` folder.
