
import torch.nn as nn


class CSGainAndBandwidthManually(nn.Module):
    def __init__(self, parameter_count=2, output_count=2):
        super(CSGainAndBandwidthManually, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(parameter_count, 30),
            nn.GELU(),
            nn.Linear(30, 60),
            nn.GELU(),
            nn.Linear(60, 120),
            nn.GELU(),
            nn.Linear(120, 300),
            nn.GELU(),
            nn.Linear(300, 300),
            nn.GELU(),
            nn.Linear(300, 120),
            nn.GELU(),
            nn.Linear(120, 60),
            nn.GELU(),
            nn.Linear(60, 30),
            nn.GELU(),
            nn.Linear(30, output_count)
        )

    def forward(self, x):
        return self.network(x)
    
class Shallow_Net(nn.Module):
    def __init__(self, parameter_count=2, output_count=2):
        super(Shallow_Net, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(parameter_count, 100),
            nn.Sigmoid(),
            nn.Linear(100, output_count),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)
    
class Sigmoid_Net(nn.Module):
    def __init__(self, parameter_count=2, output_count=2):
        super(Sigmoid_Net, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(parameter_count, 20),
            nn.Sigmoid(),
            nn.Linear(20, 20),
            nn.Sigmoid(),
            nn.Linear(20, 20),
            nn.Sigmoid(),
            nn.Linear(20, 20),
            nn.Sigmoid(),
            nn.Linear(20, 20),
            nn.Sigmoid(),
            nn.Linear(20, 20),
            nn.Sigmoid(),
            nn.Linear(20, 20),
            nn.Sigmoid(),
            nn.Linear(20, output_count),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)
class WideModel(nn.Module):
    def __init__(self, parameter_count=2, output_count=2):
        super(WideModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(parameter_count, 200),
            nn.GELU(),
            nn.Linear(200, 2000),
            nn.GELU(),
            nn.Linear(2000, 2000),
            nn.GELU(),
            nn.Linear(2000, 200),
            nn.GELU(),
            nn.Linear(200, output_count)
        )

    def forward(self, x):
        return self.network(x)

class DeepModel(nn.Module):
    def __init__(self, parameter_count=2, output_count=2):
        super(DeepModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(parameter_count, 60),
            nn.GELU(),
            nn.Linear(60, 60),
            nn.GELU(),
            nn.Linear(60, 60),
            nn.GELU(),
            nn.Linear(60, 60),
            nn.GELU(),
            nn.Linear(60, 60),
            nn.GELU(),
            nn.Linear(60, 60),
            nn.GELU(),
            nn.Linear(60, 60),
            nn.GELU(),
            nn.Linear(60, 60),
            nn.GELU(),
            nn.Linear(60, 60),
            nn.GELU(),
            nn.Linear(60, 60),
            nn.GELU(),
            nn.Linear(60, 60),
            nn.GELU(),
            nn.Linear(60, 60),
            nn.GELU(),
            nn.Linear(60, 60),
            nn.GELU(),
            nn.Linear(60, 60),
            nn.GELU(),
            nn.Linear(60, 60),
            nn.GELU(),
            nn.Linear(60, 60),
            nn.GELU(),
            nn.Linear(60, 60),
            nn.GELU(),
            nn.Linear(60, 60),
            nn.GELU(),
            nn.Linear(60, 60),
            nn.GELU(),
            nn.Linear(60, 60),
            nn.GELU(),
            nn.Linear(60, 60),
            nn.GELU(),
            nn.Linear(60, output_count)
        )

    def forward(self, x):
        return self.network(x)
    
class Model500GELU(nn.Module):
    def __init__(self, parameter_count=2, output_count=2):
        super(Model500GELU, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(parameter_count, 200),
            nn.GELU(),
            nn.Linear(200, 300),
            nn.GELU(),
            nn.Linear(300, 500),
            nn.GELU(),
            nn.Linear(500, 500),
            nn.GELU(),
            nn.Linear(500, 300),
            nn.GELU(),
            nn.Linear(300, 200),
            nn.GELU(),
            nn.Linear(200, output_count)
        )

    def forward(self, x):
        return self.network(x)
    
class Model500SiLU(nn.Module):
    def __init__(self, parameter_count=2, output_count=2):
        super(Model500SiLU, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(parameter_count, 200),
            nn.SiLU(),
            nn.Linear(200, 300),
            nn.SiLU(),
            nn.Linear(300, 500),
            nn.SiLU(),
            nn.Linear(500, 500),
            nn.SiLU(),
            nn.Linear(500, 300),
            nn.SiLU(),
            nn.Linear(300, 200),
            nn.SiLU(),
            nn.Linear(200, output_count)
        )

    def forward(self, x):
        return self.network(x)
    
class Model500Tan(nn.Module):
    def __init__(self, parameter_count=2, output_count=2):
        super(Model500Tan, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(parameter_count, 200),
            nn.Tanh(),
            nn.Linear(200, 300),
            nn.Tanh(),
            nn.Linear(300, 500),
            nn.Tanh(),
            nn.Linear(500, 500),
            nn.Tanh(),
            nn.Linear(500, 300),
            nn.Tanh(),
            nn.Linear(300, 200),
            nn.Tanh(),
            nn.Linear(200, output_count)
        )

    def forward(self, x):
        return self.network(x)
    
class DeepModel2(nn.Module):
    def __init__(self, input_size=2, output_size=2):
        super(DeepModel2, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 30),
            nn.GELU(),
            nn.Linear(30, 60),
            nn.GELU(),
            nn.Linear(60, 120),
            nn.GELU(),
            nn.Linear(120, 300),
            nn.GELU(),
            nn.Linear(300, 300),
            nn.GELU(),
            nn.Linear(300, 300),
            nn.GELU(),
            nn.Linear(300, 300),
            nn.GELU(),
            nn.Linear(300, 300),
            nn.GELU(),
            nn.Linear(300, 300),
            nn.GELU(),
            nn.Linear(300, 300),
            nn.GELU(),
            nn.Linear(300, 300),
            nn.GELU(),
            nn.Linear(300, 300),
            nn.GELU(),
            nn.Linear(300, 120),
            nn.GELU(),
            nn.Linear(120, 60),
            nn.GELU(),
            nn.Linear(60, 30),
            nn.GELU(),
            nn.Linear(30, output_size)
        )

    def forward(self, x):
        return self.network(x)
        
        






