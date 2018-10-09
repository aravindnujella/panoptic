class fit:
    def __init__(self, model, optimizer, loss_fn, data_loader, steps_per_epoch):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.data_loader = data_loader

        self.step = 0
        self.steps_per_epoch = steps_per_epoch
        self.running_loss = 0
    
    def fit(self):
        for i, data in enumerate(data_loader, 0):
            x = data
            
            self.optimizer.zero_grad()
            
            y = self.model(x)
            l = self.loss_fn(x)

            l.backward()
            optimizer.step()

            self.display_loss(l)

    def display_loss(self, l):
        self.step += 1
        self.running_loss += l
        if self.step % self.steps_per_epoch == 0:
            print(self.running_loss/self.step)
            self.step = 0

