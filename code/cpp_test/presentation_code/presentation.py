import torch

model = torch.nn.Linear(num_features, num_classes)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
data_loader = torch.utils.data.DataLoader(dataset)

for epoch in range(num_epochs):
    for input_data, label in data_loader:
        prediction = model.forward(input_data)
        loss = loss_function(prediction, label)
        loss.backward()
        optimizer.step()
