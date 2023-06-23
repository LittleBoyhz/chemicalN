import pkg_resources
import torch
from rxnfp.models import SmilesClassificationModel
import pandas as pd

from rxn_yields.data import generate_buchwald_hartwig_rxns
df = pd.read_excel('./rxn_yields/data/Buchwald-Hartwig/Dreher_and_Doyle_input_data.xlsx', sheet_name='FullCV_01')
df['rxn'] = generate_buchwald_hartwig_rxns(df)

train_df = df.iloc[:2767][['rxn', 'Output']] 
test_df = df.iloc[2767:][['rxn', 'Output']] #

train_df.columns = ['text', 'labels']
test_df.columns = ['text', 'labels']
mean = train_df.labels.mean()
std = train_df.labels.std()
train_df['labels'] = (train_df['labels'] - mean) / std
test_df['labels'] = (test_df['labels'] - mean) / std
train_df.head()
model_args = {
     'num_train_epochs': 15, 'overwrite_output_dir': True,
    'learning_rate': 0.00009659, 'gradient_accumulation_steps': 1,
    'regression': True, "num_labels":1, "fp16": False,
    "evaluate_during_training": False, 'manual_seed': 42,
    "max_seq_length": 300, "train_batch_size": 16,"warmup_ratio": 0.00,
    "config" : { 'hidden_dropout_prob': 0.7987 } 
}
model_path =  pkg_resources.resource_filename(
                "rxnfp",
                f"models/transformers/bert_pretrained" # change pretrained to ft to start from the other base model
)

yield_bert = SmilesClassificationModel("bert", model_path, num_labels=1, 
                                       args=model_args, use_cuda=torch.cuda.is_available())

yield_bert.train_model(train_df, output_dir=f"outputs_buchwald_hartwig_test_project", eval_df=test_df)

yield_predicted = yield_bert.predict(test_df.head(10).text.values)[0]
yield_predicted = yield_predicted * std + mean

yield_true = test_df.head(10).labels.values
yield_true = yield_true * std + mean

for rxn, pred, true in zip(test_df.head(10).text.values, yield_predicted, yield_true):
    print(rxn)
    print(f"predicted {pred:.1f} | {true:.1f} true yield")
    print()