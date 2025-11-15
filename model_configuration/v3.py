device = 'cuda' if torch.cuda.is_available() else 'cpu'

lr = 0.1
model = nn.Sequential(nn.Linear(1, 1)).to(device)
optimizer = optim.SGD(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

train_step_fn = make_train_step_fn(model, loss_fn, optimizer)
val_step_fn = make_val_step_fn(model, loss_fn)

writer = SummaryWriter('runs/simple_linear_regression')
x_dummy, y_dummy = next(iter(train_loader))
writer.add_graph(model, x_dummy.to(device))
