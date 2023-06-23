'''
import dgl
import torch
import torch.nn as nn

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, out_feats, num_hidden_layers, aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(SAGEConv(in_feats, out_feats, aggregator_type))
        # hidden layers
        for i in range(num_hidden_layers - 1):
            self.layers.append(SAGEConv(out_feats, out_feats, aggregator_type))
        # output layer
        self.layers.append(SAGEConv(out_feats, out_feats, aggregator_type))

    def forward(self, g, input_features):
        h = input_features
        for layer in self.layers:
            h = layer(g, h)
        return h

class MolecularPropertyPredictor(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(MolecularPropertyPredictor, self).__init__()
        self.gnn = GraphSAGE(in_feats, hidden_size, 2, 'mean')
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, g, input_features):
        h = self.gnn(g, input_features)
        logits = self.fc(h)
        return logits
# define model, optimizer, and loss function
model = MolecularPropertyPredictor(in_feats, hidden_size, num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

# training loop
for epoch in range(num_epochs):
    for g, features, labels in train_data:
        logits = model(g, features)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# evaluation loop
for g, features, labels in test_data:
    logits = model(g, features)
    _, indices = torch.max(logits, dim=1)
    correct += torch.sum(indices == labels)
    total += len(labels)
print(f'Accuracy: {correct/total:.4f}')
def generate_smile_h_dict(smiles):
    dict = {}
'''