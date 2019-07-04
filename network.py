import torch

class Policy(torch.nn.Module):
    def __init__(self, c_in, actions_out):
        super(Policy, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(c_in, 32, 3, stride=2, padding=1, bias=False),
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(32,32, 3, stride=2, padding=1, bias=False),
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(32,32, 3, stride=2, padding=1, bias=False),
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(32,32, 3, stride=2, padding=1, bias=False),
            torch.nn.ReLU()
        )

        # Assuming input size of 42x42, the output of the convolution becomes 32x3x3 = 288
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(288, 256),
            torch.nn.ReLU()
        )
        
        self.actor_fc = torch.nn.Linear(256, actions_out)
        self.critic_fc = torch.nn.Linear(256, 1)
        

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        logit = self.actor_fc(x)
        value = self.critic_fc(x)

        return logit, value


# To get random features, create a FeatureEncoder instance and use it `with torch.no_grad()`
class FeatureEncoder(torch.nn.Module):
    def __init__(self, c_in):
        super(FeatureEncoder, self).__init__()
        # input size is supposed to be (batch_size, 4, 42, 42)
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(c_in, 32, 3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(32,32, 3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(32,32, 3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(32,32, 3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.num_features = 288
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return x # (batch_size, 288)


class InverseModel(torch.nn.Module):
    def __init__(self, num_features, num_actions):
        super(InverseModel, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(num_features * 2, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_actions),
        )
        
    def forward(self, x_cur, x_next):
        # each input is supposed to be (batch_size, feature_size)
        x = torch.cat([x_cur, x_next], dim=1)
        x = self.fc(x)
        x = torch.softmax(x, dim=1)
        return x


class ForwardModel(torch.nn.Module):
    def __init__(self, num_features, num_actions):
        super(ForwardModel, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(num_features + num_actions, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 288)
        )
        
    def forward(self, x, action):
        # x: (batch_size, 288), action: (batch_size, num_actions)
        x = torch.cat([x, action], dim=1)
        x = self.fc(x)
        return x


class IntrinsicCuriosityModule(torch.nn.Module):
    def __init__(self, c_in, num_actions, use_random_features):
        super(IntrinsicCuriosityModule, self).__init__()
        self.random = use_random_features
        self.encoder = FeatureEncoder(c_in)
        if self.random:
            self.encoder.eval()
        else:
            self.inv_model = InverseModel(self.encoder.num_features, num_actions)
        self.fwd_model = ForwardModel(self.encoder.num_features, num_actions)

    def forward(self, x0, a, x1):
        # x0, x1: (batch_size, 4, 42, 42), a: (batch_size, num_actions)
        with torch.set_grad_enabled(not self.random):
            s0 = self.encoder(x0)
            s1 = self.encoder(x1)
        action_pred = self.inv_model(s0, s1) if not self.random else None
        s1_pred = self.fwd_model(s0, a)
        return s1, s1_pred, action_pred
