import sys
import torch
from torch import nn
import pandas as pd

# importing dataset
df = pd.read_csv("loan_data.csv")
df = df[["loan_status", "person_income", "loan_intent", "loan_percent_income"]]

# one-hot encodding the object type column
df = pd.get_dummies(df, columns=["loan_intent"])

# validation and training split
df_train = df.sample(frac=0.8, random_state=42) # 80% Dataset for Training
df_val = df.drop(index=df_train.index)


x_data = df_train.drop("loan_status", axis=1).astype('float32').values
x_train = torch.tensor(x_data, dtype=torch.float32)
print("Training Input Data Shape: ",x_train.shape)
x_mean = x_train.mean(axis=0) # Normalizing the data set
x_std = x_train.std(axis=0)
x_train = (x_train - x_mean) / x_std

y_train = torch.tensor(df_train["loan_status"].values, dtype=torch.float32).reshape((-1,1))
print("Training Output Data Shape: ",y_train.shape)


x_val_data = df_val.drop("loan_status", axis=1).astype('float32').values
x_val = torch.tensor(x_val_data, dtype=torch.float32)
print("Training Input Data Shape: ",x_val.shape)
x_mean = x_val.mean(axis=0)
x_std = x_val.std(axis=0)
x_val = (x_val - x_mean) / x_std

y_val = torch.tensor(df_val["loan_status"].values, dtype=torch.float32).reshape((-1,1))
print("Training Output Data Shape: ",y_val.shape)

# creating model
model = nn.Sequential(
    nn.Linear(8,32),
    nn.ReLU(),
    nn.Linear(32,16),
    nn.ReLU(),
    nn.Linear(16,1)
)

loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.002)

# initialising batch sizes
num_entries = x_train.size(0)
batch_size = 2048

# training pass
for i in range (0,20000):
    loss_sum=0
    for start in range(0,num_entries,batch_size):
        end = min(num_entries, start + batch_size)
        x_data = x_train[start:end]
        y_data = y_train[start:end]

        optimizer.zero_grad()
        output = model(x_train)
        loss = loss_fn(output, y_train)
        loss.backward()
        loss_sum += loss.item()
        optimizer.step()

    if i%100 == 0:
        print(i)
        print(loss_sum)

# Model Evaluation
model.eval()
with torch.no_grad():
    print("Validation")
    outputs = model(x_val)
    y_pred = nn.functional.sigmoid(outputs) > 0.5
    y_pred_correct = y_pred.type(torch.float32) == y_val

    print("Accuracy: ", (y_pred == y_val).type(torch.float32).mean())
    print("Sensitivity: ", (y_pred[y_val==1] == y_val[y_val==1]).type(torch.float32).mean())
    print("Specificity: ", (y_pred[y_val==0] == y_val[y_val==0]).type(torch.float32).mean())
    print("Precision: ", (y_pred[y_pred==1] == y_val[y_pred==1]).type(torch.float32).mean())

    print(y_pred_correct.type(torch.float32).mean())

    print("Training")
    outputs = model(x_train)
    y_pred = nn.functional.sigmoid(outputs) > 0.5
    y_pred_correct = y_pred.type(torch.float32) == y_train

    print("Accuracy: ", (y_pred == y_train).type(torch.float32).mean())
    print("Sensitivity: ", (y_pred[y_train==1] == y_train[y_train==1]).type(torch.float32).mean())
    print("Specificity: ", (y_pred[y_train==0] == y_train[y_train==0]).type(torch.float32).mean())
    print("Precision: ", (y_pred[y_pred==1] == y_train[y_pred==1]).type(torch.float32).mean())

