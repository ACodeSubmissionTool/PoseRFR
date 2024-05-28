
## The devil is in the details: Delving into large-pose blind face restoration



## <a name="installation"></a>:crossed_swords:Installation

```shell
pip install -r requirements.txt
```

## <a name="pretrained_models"></a>:dna:Pretrained Models And Datasets

The pretrained model weights, test_set and partial tranining set from [FFHQ]() can be found on the [BaiduYun](https://pan.baidu.com/s/18PL5HPORbFaQ4am9KD_l9A?pwd=fxf4).


## <a name="inference"></a>:crossed_swords:Inference


We prepare some examples to show how to do inference.

<a name="inference_fr"></a>: Stage I Inference

```shell
python inference_face.py  --config ./configs/model/cldm_twoS.yaml  --input input_path --ckpt ./checkpoints/stage1.ckpt --output output_path --device cuda 
```


<a name="inference_fr"></a>: Stage II Inference

```shell
python inference_face.py  --config ./configs/model/cldm_twoS_wD.yaml  --input input_path --ckpt ./checkpoints/stage2.ckpt --output output_path   --device cuda 

```


<a name="inference_fr"></a>: Stage III Inference

```shell
python inference_face.py  --config ./configs/model/cldm_twoS_AFR.yaml  --use_afr --input input_path  --ckpt ./checkpoints/stage2.ckpt --output output_path   --device cuda 
```


## <a name="train"></a>:stars:Train


1. Generate file list of training set and validation set, a file list looks like:

    ```txt
    /path/to/image_1
    /path/to/image_2
    /path/to/image_3
    ```
    
    We prepare a script for you to generate the file list: [make_file_list](make_file_list.py). 

    ```shell
    python make_file_list.py --img_folder ./data_folder --save_path ./test.list 
    
    ```
2. Fill in the train and val configuration file with yout file list script: [face*.yaml](./configs). 

3. training stage I

    ```shell
    HF_ENDPOINT=https://hf-mirror.com python train.py --config ./configs/train_stage1.yaml
    ```
4. training stage II

    ```shell
    HF_ENDPOINT=https://hf-mirror.com python train.py --config ./configs/train_stage2.yaml
    ```

5. generate latent code to reduce computation cost:

    ```shell
    python inference_face.py  --config ./configs/model/cldm_twoS_wD.yaml --generate_latent  --ckpt ./checkpoints/stage2.ckpt --output ./results/test_generate  --device cuda 
    ```
    
    This code will generate the latent code to ' ./results/test_generate ', please use [make_file_list](make_file_list.py) to generate file list and replace the path in  [stage3*.yaml](./configs/)
6. training stage III

    ```shell
    HF_ENDPOINT=https://hf-mirror.com python train.py --config ./configs/train_stage3.yaml
    ```




## Acknowledgement

This project is based on [ControlNet](https://github.com/lllyasviel/ControlNet), [BasicSR](https://github.com/XPixelGroup/BasicSR), [StableSR](https://github.com/IceClear/StableSR), [DiffBIR](https://github.com/XPixelGroup/DiffBIR), [GFPGAN](https://github.com/TencentARC/GFPGAN/tree/master/gfpgan), [DifFace](https://github.com/zsyOAOA/DifFace). Thanks for their awesome work.

