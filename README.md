# Recurrent Diacritizer

This repository contains the code for our paper on Arabic diacritization:

"Arabic Diacritization with Recurrent Neural Networks", Yonatan Belinkov and James Glass, EMNLP 2015.

## Requirements
* Cuda-enabled GPU card
* [Currennt](http://sourceforge.net/projects/currennt)
* Python libraries: numpy, sklearn, matplotlib, netCDF4, gensim
* This code was tested with Ubuntu 12.04 and 14.04. 

## Data preparation
1. The input Arabic text must be given in [Buckwalter transliteration](http://www.qamus.org/transliteration.htm). Each text must have two files, with and without diacritics, containing one word on each line. An empty line separates sentences. 

2. To prepare the data for training, run `prepare_data_for_currennt.py`. You have to specify train and test sets, and can optionally specify a development set. The following is an example command:
```
python prepare_data_for_currennt.py -twf train.txt -twdf train_diac.txt -tncf train.nc \ 
-swf test.txt -swdf test_diac.txt -sncf test.nc \ 
-dwf dev.txt -dwdf dev_diac.txt -dncf dev.nc
```
This will produce train, dev, and test files in netCDF format. For more options, do: `python prepare_data_for_currennt -h`.

## Training
1. Edit `run-con.sh` and point /path/to/currennt to your currennt binary file. Also define the Cuda device you want to use with CURRENNT_CUDA_DEVICE. 
2. Edit config.cfg and define the paths to your train, dev, and test files. These are the netCDF `.nc` files created previously. Also take note of the saved model file name defined under "save_network".
3. (Optional) Edit config.cfg to change some of the training parameters. You may also want to change the default network architecture by pointing to your own network.jsn file.
4. run `sh run-con.sh`.

## Testing
After training, you will have a trained model, say network-final.jsn. To generate predictions, do:
```
/path/to/currennt --network network-final.jsn --ff_input_file test.nc --ff_output_file test.csv.out --revert_std false
```
This will run the trained model in network-final.jsn on the test file test.nc and save the predictions in test.csv.out. 

To evaluate the predictions, do:
```
python eval_currennt.py test.nc test.csv.out train.nc
```
This will print some statistics including diacritic error rate (DER). Note that Currennt will print some error rates during and after training, but those may not be accurate, as they include a word boundary symbol. 

To write the diacrizied text, do:
```
python write_currennt_predictions.py test.txt test_diac.txt test.csv.out pred.txt train.nc
```
The predicted diacritized text will be written to pred.txt. 

### Diacritizing raw text
To diacritize a raw text, you'll need to prepare it in the format specified under **Data preparation**. Then, convert the file to netCDF format using `prepare_data_for_currennt.py`. 

You can use the final trained 3-layer B-LSTM model described in our paper. The trained weights are available in the `*-final.jsn` file. 

You will want to make sure the input features (letter vectors) and label indices in the netCDF file you create are identical to those used in the trained model. For this, you can provide `letter_vectors.txt` and `label_indices.txt` correspondingly. Consult the options in `python prepare_data_for_currennt -h` for passing these as arguments. 

## Citing
If you use this code in your work, please consider citing our paper:
"Arabic Diacritization with Recurrent Neural Networks", Yonatan Belinkov and James Glass, EMNLP 2015.

```bib
@InProceedings{belinkov-glass:2015:EMNLP,
  author    = {Belinkov, Yonatan  and  Glass, James},
  title     = {Arabic Diacritization with Recurrent Neural Networks},
  booktitle = {Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing},
  month     = {September},
  year      = {2015},
  address   = {Lisbon, Portugal},
  publisher = {Association for Computational Linguistics},
  pages     = {2281--2285},
  url       = {http://aclweb.org/anthology/D15-1274}
}
```

## Questions?
For any questions or suggestions, email belinkov@mit.edu. 

## TODO
* Provide data.
