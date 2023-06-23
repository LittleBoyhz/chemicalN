# 获取训练集和测试集
cd ./molecular/data
python split.py
cp -f real_4_smiles_yields_product_2_train_2.xlsx ../../datasets/real/real_4/
cp -f real_4_smiles_yields_product_2_test.xlsx ../../datasets/real/real_4/
cd ../..

# 使用训练集拓展指定数据
cd ./utils
python get_train_2.py
python get_test.py
cd ..
python VAE.py

#cd ./utils
#python get_unlabel.py
#cd ..
#python extend.py #python add_excel.py
# 完成测试
#cd ./utils
#python get_train_2.py
#python get_test.py
#cd ..
#python VAE.py
