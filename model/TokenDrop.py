import torch
import torch.nn as nn

class TokenDrop(nn.Module):
    """For a batch of tokens indices, randomly replace a non-specical token with <pad>.
    
    Args:
        prob (float): probability of dropping a token
        pad_token (int): index for the <pad> token
        num_special (int): Number of special tokens, assumed to be at the start of the vocab
    """

    def __init__(self, prob=0.1, pad_token=0, num_special=4):
        self.prob = prob
        self.num_special = num_special
        self.pad_token = pad_token

    def __call__(self, sample):
        mask = torch.bernoulli(self.prob * torch.ones_like(sample)).long()
        
        can_drop = (sample >= self.num_special).long()
        mask = mask * can_drop
        
        replace_with = (self.pad_token * torch.ones_like(sample)).long()
        
        sample_out = (1 - mask) * sample + mask * replace_with
        
        return sample_out

class TokenDropEven(nn.Module):
    """For a batch of tokens indices, randomly replace a non-special token with <pad>.
    
    Args:
        prob (float): probability of dropping a token
        pad_token (int): index for the <pad> token
        num_special (int): Number of special tokens, assumed to be at the start of the vocab
    """

    def __init__(self, prob=0.1, pad_token=0, num_special=4):
        super().__init__()
        self.prob = prob
        self.num_special = num_special
        self.pad_token = pad_token

    def __call__(self, sample):
        """Mask only action (even if it is not a special token) for a batch of samples."""
        sample_out = torch.empty_like(sample)
        for i, single_sample in enumerate(sample):
            even_sample = single_sample[::2]
            odd_sample = single_sample[1::2]
            
            # Generate mask for even elements
            mask = torch.bernoulli(self.prob * torch.ones_like(even_sample)).long()
            can_drop = (even_sample >= self.num_special).long()
            mask = mask * can_drop
            
            # Replace masked elements with <pad>
            replace_with = (self.pad_token * torch.ones_like(even_sample)).long()
            even_sample_out = (1 - mask) * even_sample + mask * replace_with
            
            # Recombine even and odd elements
            sample_out[i, ::2] = even_sample_out
            sample_out[i, 1::2] = odd_sample
        
        return sample_out

class TokenDropOdd(nn.Module):
    """For a batch of tokens indices, randomly replace a non-special token with <pad>.
    
    Args:
        prob (float): probability of dropping a token
        pad_token (int): index for the <pad> token
        num_special (int): Number of special tokens, assumed to be at the start of the vocab
    """

    def __init__(self, prob=0.1, pad_token=0, num_special=4):
        super().__init__()
        self.prob = prob
        self.num_special = num_special
        self.pad_token = pad_token

    def __call__(self, sample):
        """Mask only action (even if it is not a special token) for a batch of samples."""
        sample_out = torch.empty_like(sample)
        for i, single_sample in enumerate(sample):
            even_sample = single_sample[::2]
            odd_sample = single_sample[1::2]
            
            # Generate mask for odd elements
            mask = torch.bernoulli(self.prob * torch.ones_like(odd_sample)).long()
            can_drop = (odd_sample >= self.num_special).long()
            mask = mask * can_drop
            
            # Replace masked elements with <pad>
            replace_with = (self.pad_token * torch.ones_like(odd_sample)).long()
            old_sample_out = (1 - mask) * odd_sample + mask * replace_with
            
            # Recombine even and odd elements
            sample_out[i, ::2] = even_sample 
            sample_out[i, 1::2] = old_sample_out
        
        return sample_out
    
class TokenDropOddWithOH(nn.Module):
    """For a batch of tokens indices, randomly replace a non-special token with <pad>.
    
    Args:
        prob (float): probability of dropping a token
        pad_token (int): index for the <pad> token
        num_special (int): Number of special tokens, assumed to be at the start of the vocab
    """

    def __init__(self, oh, prob=0.1, mask_token=-1):
        super().__init__()
        self.prob = prob
        self.size_oh = oh.length
        self.mask_token = mask_token

    def __call__(self, sample):
        """Mask only action (even if it is not a special token) for a batch of samples."""
        assert sample.shape[1] % self.size_oh == 0, "The input sample must be divisible by size_oh."
        matrix_to_mask = torch.zeros(sample.shape[1] // self.size_oh, dtype=torch.int32)
        sample_out = torch.empty_like(sample)
        for i, single_sample in enumerate(sample):
            even_sample = matrix_to_mask[::2]
            odd_sample = matrix_to_mask[1::2]
            odd_sample = torch.bernoulli(self.prob * torch.ones_like(odd_sample)).long()
            # mask = remake matrix_to_mask
            stacked = torch.cat((even_sample.unsqueeze(1), odd_sample.unsqueeze(1)), dim=1)
            mask = stacked.flatten()
            mask = torch.repeat_interleave(mask, self.size_oh)
            mask = mask.byte()
            single_sample = single_sample.masked_fill(mask.bool(), -1)
            sample_out[i] = single_sample       
        
        return sample_out
