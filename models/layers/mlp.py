from torch import nn


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.1,
    ):
        """
        Args:
            in_features: The size of the input features.
            hidden_features: The size of the hidden features. Defaults to None, which sets it to the size of the input features.
            out_features: The size of the output features. Defaults to None, which sets it to the size of the input features.
            act_layer: The activation function layer to use. Defaults to nn.GELU.
            drop: The dropout probability to use. Defaults to 0.1.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
