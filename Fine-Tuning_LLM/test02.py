import torch


class Net(torch.nn.Module):
    
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.emb = torch.nn.Embedding(20000,128)
        self.emb2 = torch.nn.Embedding(5,128)
        
        self.tf_layer = None
        
        self.output_layer = None
        
    def forward(self,ids):
        ids2 = torch.tensor([0,1,2,3,4,5])[None]
        token2 = self.emb2(ids2)
        
        with torch.no_grad():
            tokens = self.emb(ids)
            new_tokens = torch.cat([token2,tokens],dim=1)
            feature = self.tf_layer(new_tokens)
            out = self.output_layer(feature)
        
        return out
    
    



