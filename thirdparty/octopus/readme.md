## Octopus Setup/Installation

Download and install DIRT: https://github.com/pmh47/dirt.

Download the neutral SMPL model from http://smplify.is.tue.mpg.de/, rename it from "basicModel_neutral_..." to neutral_smpl.pkl and place it in the `assets` folder.
```
# or you can use the terminal
cp <path_to_smplify>/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl assets/neutral_smpl.pkl
```

Download pre-trained model weights from [here](https://drive.google.com/open?id=1_CwZo4i48t1TxIlIuUX3JDo6K7QdYI5r) and place them in the `assets` folder.

```
# or you can use the terminal
unzip <downloads_folder>/octopus_weights.hdf5.zip -d assets/
```

