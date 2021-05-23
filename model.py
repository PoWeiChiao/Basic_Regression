import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(Net, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_feature, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_feature)
        )

    def forward(self, x):
        return self.seq(x)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Net(10, 1)
    net.to(device=device)

    input = torch.randn(1, 10)
    input = input.to(device=device)
    print(input)
    output = net(input)
    print(output)

if __name__ == '__main__':
    main()