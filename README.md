# Evaluate
主体代码在evaluate.py中    
  
config文件格式有所改动，具体参考config_template.yaml,通俗易懂只需要传入少量参数。  
  
更改测试模型与数据集，只需更改model.checkpoint,dataset.name,dateset.data_dir,请指定好test.output_dir,规范路径格式,防止与别的数据集以及模型搞混,一般格式如下:   
  
'./experiments/模型名称(指明训练数据集,epoch和model_name)/测试数据集(EVE, XGAZE, GAZE360, MPII, etc.)'  
  


## Dataset
建议的数据格式如下：  

--data  
    --MPII  
        --00.h5  
        --01.h5  
        ...  
    --GAZE360  
        --test  
            --24.h5  
            --30.h5  
            ...  
    --XGAZE  
        --train_test_split.json
        --train
            ...
        --test
            --00.h5
            --01.h5
            ...
    --EVE  
        ...  
    ...  

## MPII
注意！MPIIFaceGaze需要传入测试集的person id来指定哪个人的数据作为测试集，修改test.test_id(0~14)  

## GAZE360
GAZE360的文件路径是一个包含test文件夹的文件夹   

## XGAZE
XGAZE的测试集不包含真实的gaze label,测试XGAZE时，evaluate.py将会在结果文件夹中产生'within_eva_xxx.txt'预测结果文件，用于上传至网络平台进行测试      
XGAZE的测试基本基于源代码，在数据文件夹中需要原有的train_test_split.json文件，用于得到test的索引。
## EVE
...  

## test
主体代码在evaluate.py中,具体实现依靠Class GazeTest.  
'python evaluate.py --config /path/to/your config.yaml'