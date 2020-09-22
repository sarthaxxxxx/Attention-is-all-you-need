import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np


class SelfAttention(nn.Module):
    def __init__(self, embedding_size, heads):

        super(SelfAttention,self).__init__()
        self.embedding_size=embedding_size
        self.heads=heads
        self.head_size=int(embedding_size/heads)

        assert (self.head_size*heads==embedding_size),"Size mismatch between embedding size and number of heads (should be divisible)"

        self.query=nn.Linear(self.head_size,self.head_size,bias=False)
        self.keys=nn.Linear(self.head_size,self.head_size,bias=False)
        self.values=nn.Linear(self.head_size,self.head_size,bias=False)

        self.out=nn.Linear(self.head_size*heads,embedding_size,bias=False)


    def forward(self,query,key,value, mask=None):

        N_training_examples=query.shape[0]

        '''
        Splitting the embedding into head number of pieces.
        '''
        queries=query.reshape(N_training_examples,query.shape[1],self.heads,self.head_size)
        values=value.reshape(N_training_examples,value.shape[1],self.heads,self.head_size)
        keys=key.reshape(N_training_examples,key.shape[1],self.heads,self.head_size)

        queries=self.query(queries)
        values=self.values(values)
        keys=self.keys(keys)
         
        '''
        Output of the multiplication between query and key.
        '''
        qk=torch.einsum('nqhd,nkhd->nhqk',[queries,keys])   #Shape:(N_training_examples,head_size,query.shape[1],key.shape[1])

        if mask is not None:
            qk=qk.masked_fill_(mask==0,float('-1e40'))  #provided an extremely small value.

        attention=nn.Softmax(qk/np.sqrt(self.embedding_size),dim=-1)

        after_attention=torch.einsum('nhqk,nkhd->nqhd',[attention,values]).reshape((N_training_examples,query.shape[1],self.heads*self.head_size))
        #Attention shape : (N,head,query.shape[1],key.shape[1]) #Value shape : (N,value.shape[1],heads,head_size)

        return self.out(after_attention)


class EncoderBlock(nn.Module):
    def __init__(self,embedding_size,heads,dropout,forward_exp):
        super(EncoderBlock,self).__init__()
        self.attention=SelfAttention(embedding_size,heads)
        self.dropout=nn.Dropout(dropout)
        self.layer_normalisation=nn.LayerNorm(embedding_size)  #Performs normnalisation per training example, rather than taking an average across the entire batch. 
        self.feed_forward=nn.Sequential(nn.Linear(embedding_size,embedding_size*forward_exp),
                                        nn.ReLU(),
                                        nn.Linear(forward_exp*embedding_size,embedding_size))
    
    def forward(self, query, key, value, mask):
        attention=self.attention(query,key,value,mask)
        t=self.dropout(self.layer_normalisation(attention+query))
        ff=self.feed_forward(t)
        out=self.dropout(self.layer_normalisation(t+ff))
        return out


class Encoder(nn.Module):
    def __init__(self, embedding_size, input_vocab_size, number_layers, heads, device, forward_exp, max_length, dropout):
        '''
        max_length : related to the positional embedding. Length of the max input.
        '''
        super(Encoder,self).__init__()
        self.embedding_size=embedding_size
        self.device=device
        self.word_embedding=nn.Embedding(input_vocab_size, embedding_size)
        self.positional_embedding=nn.Embedding(max_length,embedding_size)
        self.layers=nn.ModuleList([EncoderBlock(embedding_size,heads,dropout=dropout,forward_exp=forward_exp) for _ in range(number_layers)])
        self.dropout=nn.Dropout(dropout)


    def forward(self,t,mask):
        N_training_examples,sequence_length,_= t.shape
        positions=torch.arange(0,sequence_length).expand(N_training_examples,sequence_length).to(device=self.device)

        added_output=self.dropout(self.positional_embedding(positions)+self.word_embedding(t))

        for layers in self.layers:
            out=layers(added_output,added_output,added_output,mask)
        return out


class DecoderBlock(nn.Module):
    def __init__(self,embedding_size,heads,device,forward_exp,dropout):
        super(DecoderBlock,self).__init__()
        self.attention=SelfAttention(embedding_size,heads)
        self.layer_normalisation=nn.LayerNorm(embedding_size)
        self.transformer_module=EncoderBlock(embedding_size,heads,dropout,forward_exp)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x,value, key, source_mask, target_mask):
        '''
        Before the transformer module in the decoder section; consists of the mask.
        '''
        attention=self.attention(x,x,x,target_mask)
        query=self.dropout(self.layer_normalisation(attention+x))
        '''
        Transformer module.
        '''
        out=self.transformer_module(query, key, value, source_mask)
        return out 


class Decoder(nn.Module):
    def __init__(self, embedding_size, target_vocab_size, number_layers, heads, device, forward_exp, max_length, dropout):
        super(Decoder,self).__init__()
        self.embedding_size=embedding_size
        self.device=device
        self.dropout=nn.Dropout(dropout)
        self.word_embedding=nn.Embedding(target_vocab_size, embedding_size)
        self.positional_embedding=nn.Embedding(max_length,embedding_size)
        self.layers=nn.ModuleList([DecoderBlock(embedding_size,heads,device,forward_exp=forward_exp,dropout=dropout) for _ in range(number_layers)])
        self.fc_out=nn.Linear(embedding_size,target_vocab_size)


    def forward(self,t,encoder_out,source_mask,target_mask):
        N_training_examples,sequence_length,_= t.shape
        positions=torch.arange(0,sequence_length).expand(N_training_examples,sequence_length).to(device=self.device)
        added_output=self.dropout(self.positional_embedding(positions)+self.word_embedding(t))
        for layer in self.layers:
            x=layer(x,encoder_out,encoder_out,source_mask,target_mask)
        out=self.fc_out(x)
        return out


class Transformer(nn.Module):
    def __init__(self, input_vocab_size, target_vocab_size, source_pad_idx, target_pad_idx, embedding_size=256, number_layers=6 , forward_exp=4, heads=8, dropout=0, device='cuda', max_length=100):
        super(Transformer,self).__init__()
        self.encoder=Encoder(embedding_size,input_vocab_size,number_layers,heads,device,forward_exp,max_length,dropout)
        self.decoder=Decoder(embedding_size,target_vocab_size,number_layers,heads,device,forward_exp,max_length,dropout)
        self.device=device
        self.source_pad_idx=source_pad_idx
        self.target_pad_idx=target_pad_idx

    def source_masks(self,source):
        source_mask=(source is not self.source_pad_idx).unsqueeze(1).unsqueeze(2) 
        return source_mask.to(self.device)
    
    def target_masks(self,target):
        N,seq_length=target.shape
        target_mask=torch.tril(torch.ones((seq_length,seq_length))).expand(N,1,seq_length,seq_length)
        return target_mask.to(self.device)

    def forward(self,source,target):
        source_mask=self.source_masks(source)
        target_mask=self.source_masks(target)
        encoder_output=self.encoder(source,source_mask)
        return self.decoder(target,encoder_output,source_mask,target_mask)


def main():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    source=torch.tensor([[1,2,3,4,5,6,7,8,8,9,0],[5,3,4,6,3,2,4,6,7,4,2]]).to(device=device)
    target=torch.tensor([[2,4,3,2,1],[4,3,2,3,4]]).to(device=device)

    source_pad_idx=0
    target_pad_idx=0
    source_vocab_size=12
    target_vocab_size=12

    model=Transformer(source_vocab_size,target_vocab_size,source_pad_idx,target_pad_idx).to(device=device)
    print(model)
    return model(source,target)


if __name__=='__main__':
    main()




