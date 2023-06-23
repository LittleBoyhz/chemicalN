#!/bin/sh
cd ./molecular/data
python split.py
wait
cp -f real_6_smiles_yields_product_2_test.xlsx ../../datasets/real/real_6/
wait
cp -f real_6_smiles_yields_product_2_train.xlsx ../../datasets/real/real_6/
wait
cp -f real_6_smiles_yields_product_2_train_2.xlsx ../../datasets/real/real_6/
wait
cd ..
cd ..
echo "Starting testing..."
cd ./utils
python get_test.py
wait
python get_train_2.py
cd ..
python VAE.py
wait
