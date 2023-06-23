# 获取训练集和测试集
cd ./molecular/data
python split.py
cp -f real_6_smiles_yields_product_2_train_2.xlsx ../../datasets/real/real_6/
cp -f real_6_smiles_yields_product_2_test.xlsx ../../datasets/real/real_6/
cd ../..

# 使用训练集拓展数据
cd ./utils
python get_train_2.py
python get_unlabel.py
cd ..
python extend.py

#deal with excel

#cd ./utils
#python get_train_2.py
#python get_test.py
#cd ..
#python VAE.py

