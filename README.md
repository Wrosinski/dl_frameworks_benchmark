# Deep Learning frameworks benchmark


Code for Deep Learning frameworks benchmark.
Currently work in progress, basic benchmark with artifically generated data was made.


## Structure

-   `keras/` contains code for Keras benchmarking
-   `pytorch/` contains code for Pytorch benchmarking
-   `scripts/schedule.sh` runs set of models from both benchmarks
-   `scripts/log_extract.ipynb` notebook with logs data processing & saving
-   `scripts/*.R` R scripts for data visualization based on processed logs

## Remarks

NASNet model wasn't working with `tensorflow.keras`. It threw errors, so it was finally omitted.

Versions:
-   TF: 1.12.0
-   Keras: 2.2.4
-   Torch: 1.0.0
