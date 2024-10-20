# Evaluate
主体代码在evaluate.py中  
config文件格式有所改动，具体参考config_template.yaml,通俗易懂只需要传入少量参数。  
更改测试模型与数据集，只需更改model.checkpoint,dataset.name,dateset.data_dir,请指定好test.output_dir,规范路径格式,防止与别的数据集以及模型搞混,一般格式如下:  
'./experiments/模型名称(指明训练数据集,epoch和model_name)/测试数据集(EVE, XGAZE, GAZE360, MPII, etc.)'  

## MPII
注意！MPIIFaceGaze需要传入测试集的person id来指定哪个人的数据作为测试集，修改test.test_id(0~14)  
## GAZE360
代码已经完成,需要进行测试.  

## XGAZE
...  
## EVE
...  
## test
主体代码在evaluate.py中,具体实现依靠Class GazeTest.  
'python evaluate.py --config /path/to/your config.yaml'