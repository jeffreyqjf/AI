import torch

a = torch.ones(3,4,5,7)

a = a[None]

b = torch.randn(3, 2, 3)

c = b.sum(1)

d, e = b.max(1)
#print(d.shape, e.shape)
#print(d, "\n", e)
#  print(b[e])  ???
#print(b.unsqueeze(2).unsqueeze(2).shape)
one = torch.ones(2, 3, 5, 5,dtype=torch.double)
zeros = torch.zeros(1, 3)

#print(one.dtype)
zeros.to(dtype=torch.short)
one_t = torch.transpose(one, 2, 1)
#print(one_t.shape)
stor = torch.tensor([
    [1, 2, 3],
    [3, 4, 5]
])
stor_t = stor.transpose(1, 0)
#print(stor.storage())
#print(stor_t.storage())# id 不同 storage 一样吗
#print(stor_t)
#print(stor[1].storage_offset())


