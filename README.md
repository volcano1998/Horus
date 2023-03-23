# :milky_way: Horus :eagle: 
image CVD accessibility evaluation tool


## Installation

you need to have a python3 (>=3.7) environment. You should have below packages installed.
```
numpy
matplotlib
pandas
PIL
sklearn
statistics
scipy
opencv-python
```
You will also need to clone this github repo.
```
git clone https://github.com/volcano1998/Horus.git
```
## Usage

The pipeline Horus.py needs 3 arguments as input

-i  input image you want to evaluate
-o output folder to store the evaluation result
-m pretrained model used for prediction


You can have a test run by using the following command
cd Horus
python3 Horus.py -i sample_image/sample.jpeg -o test_run/ -m pretrained_model/finalized_model.sav

where sample_image.sample.jpeg is a image we picked from a random journal, test_run/ will be the output folder, pretrained_model/finalized_model.sav is the model we provide you to use.

If you have installed the Horus successfully, you will see 6 files under test_run folder. They are 

prediction.txt	final prediction file, which will tell you this picture is CVD friend or not

and 5 imtermidiate results

sample_og.pdf  key colors from original image
sample_og.csv	 key colors in RGB format from original image
sample_cb.jpeg  the sample image under a CVD filter(simulated image)
sample_cb.pdf	key colors from simulated image
sample_cb.csv key colors in RGB format from simulated image

The whole evaluation procedure takes about 13 seconds per picture.




