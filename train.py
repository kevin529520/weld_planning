from Model import Net, predict
from DataLoader import ForceDataset
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

class Pose2ForceTrain(object):
    def __init__(self) -> None:
        self.batch_size = 256
        self.learning_rate = 0.001
        self.momentum = 0.3
        self.epoch = 200
        self.saved_pth = "./weights/weight.pth"

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        # self.loss =torch.nn.MSELoss()
        self.data_list = ["./data/1.txt","./data/2.txt"]
 

    def train(self):
        # load data
        dataset = ForceDataset(file_list=self.data_list, data_model='train')
        print("sample numbers: ", len(dataset))
        traing_data = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False, num_workers=8, drop_last=False, pin_memory=False, collate_fn=None)
        
        force_model = Net().to(self.device)
        # optimizer
        optimizer = optim.Adam(force_model.parameters(), lr=self.learning_rate)
        vloss = torch.nn.MSELoss()

        # training
        force_model.train()
        for epoch in range(self.epoch):
            running_loss = 0.0
            for i, datax in enumerate(traing_data):
                pose= datax[0].to(self.device)
                force= datax[1].to(self.device)
                # force = force.squeeze(1)

                optimizer.zero_grad()

                outputs = force_model(pose)
                l1 = vloss(outputs[:,0:3], force[:,0:3])
                l2 = vloss(outputs[:,3:6], force[:,3:6])
                loss = 0.1*l1 + 10*l2
                running_loss += loss.item()

                loss.backward()
                optimizer.step()
                
                if i % 20 == 19:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss/20))
                    running_loss = 0.0


        # save weight
        torch.save(force_model.state_dict(), self.saved_pth)

    @torch.no_grad()
    def var(self,pre_weight="1.pth"):
        # load data
        dataset = ForceDataset(file_list=self.data_list, data_model='var')
        print("sample numbers: ", len(dataset))
        var_data = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False, num_workers=8, drop_last=False, pin_memory=False, collate_fn=None)
        
        force_model = Net().to(self.device)
        # optimizer
        optimizer = optim.Adam(force_model.parameters(), lr=self.learning_rate)
        vloss = torch.nn.MSELoss()

        # training
        force_model.eval()
        running_loss = 0.0
        for i, datax in enumerate(var_data):
            pose= datax[0].to(self.device)
            force= datax[1].to(self.device)
            # force = force.squeeze(1)

            outputs = force_model(pose)
            l1 = vloss(outputs[:,0:3], force[:,0:3])
            l2 = vloss(outputs[:,3:6], force[:,3:6])
            loss = 0.1*l1 + 10*l2
            running_loss += loss.item()


        print("var loss:", running_loss/len(var_data))
        return running_loss/len(var_data)


        


if __name__ == "__main__":
    a  = Pose2ForceTrain()
    a.train()
