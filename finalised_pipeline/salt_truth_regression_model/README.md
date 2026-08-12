### Using Truth Regression Model Output

In this project the model was trained to regress scores directly onto GN2 flavour tag probabilities. This means the model output SHOULD NOT be softmaxed. Actual mean and std values have been supplied to the model, as determined during preprocessing. The scores need to be un-normalised before analysis.
