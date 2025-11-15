n_epochs = 1000

for i in range(n_epochs):
    model.train()
    y_pred = model(x_train_tensor)
    loss = loss_fn(y_pred, y_train_tensor)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
