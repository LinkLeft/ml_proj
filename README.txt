具体开发逻辑如下：
RAWDATAPROCESSING.ipynb数据处理——计算并输出所有数据的三分类情况【classified_stock_data.xlsx】并将三分类情况添加到【GraphAutoencoderoutput_data.xlsx】中生成【GraphAutoencoderoutput_data2.0.xlsx】用于训练

GraphAutoencoder.ipynb——图自编码器将图数据转化为向量数据，输入【ticker_train_data.json】（此文件为traindata和testdata的合并数据），【train_stock_data.xlsx】（将训练出的32维向量拼接在这个文档数据后面）输出【GraphAutoencoderoutput_data.xlsx】

LSTMMLP.ipynb——LSTM+MLP三分类，输入【GraphAutoencoderoutput_data2.0.xlsx】自动分训练集测试集然后训练输出F1值