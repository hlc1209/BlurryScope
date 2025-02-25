# BlurryScope: a cost-effective and compact scanning microscope for automated HER2 scoring using deep learning on blurry image data

This repository contains codes and a subset of the testing dataset for reproducing the BlurryScope results.

## Network Training Testing Codes

To reproduce the results, clone the whole repository and prepare an environment with CUDA devices.

1. **Testing Procedure**: 

    To run the 4-class model
    ```
    cd score-4/
    python test.py
    ```
    
    To run the 2-class model
    ```
    cd score-2/
    python test.py
    ```

2. **Dependencies**: The models are implemented using Pytorch 2.6.0.

## Support

Should you encounter any problems while running the codes, please feel free to contact [fanous@g.ucla.edu](mailto:fanous@g.ucla.edu) or [hanlong@g.ucla.edu](mailto:hanlong@g.ucla.edu).

## License

This project is open-sourced under the Apache License 2.0. See the LICENSE file for details.