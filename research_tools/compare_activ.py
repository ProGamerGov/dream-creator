import torch



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-activation_one", type=str, default='tensor1.pt')    
    parser.add_argument("-activations_two", type=str, default='tensor2.pt')  
    params = parser.parse_args()
    main_func(params)


def main_func(params):
    tensor1 = torch.load(params.activation_one)
    tensor2 = torch.load(params.activation_two)

    diff_tensor = torch.abs(tensor1 - tensor2)

    get_ranked = RankChannels([5], 'strong')
    test = get_ranked(diff_tensor.unsqueeze(0))
    
    print('The most similar channels between the activation inputs are:')
    print('  ', test)


def sort_channels(input):
    channel_list = []
    for i in range(input.size(1)):
        channel_list.append(torch.mean(input.clone().squeeze(0).narrow(0,i,1)).item())
    return sorted((c,v) for v,c in enumerate(channel_list))
        

# Define an nn Module to rank channels based on activation strength
class RankChannels(torch.nn.Module):

    def __init__(self, channels=1, channel_mode='strong'):
        super(RankChannels, self).__init__()
        self.channels = channels
        self.channel_mode = channel_mode

    def sort_channels(self, input):
        channel_list = []
        for i in range(input.size(1)):
            channel_list.append(torch.mean(input.clone().squeeze(0).narrow(0,i,1)).item())
        return sorted((c,v) for v,c in enumerate(channel_list))

    def get_middle(self, sequence):
        num = self.channels[0]
        m = (len(sequence) - 1)//2 - num//2
        return sequence[m:m+num]

    def remove_channels(self, cl):
        return [c for c in cl if c[1] not in self.channels]

    def rank_channel_list(self, input):
        top_channels = self.channels[0]
        channel_list = self.sort_channels(input)

        if 'strong' in self.channel_mode:
            channel_list.reverse()
        elif 'avg' in self.channel_mode:
            channel_list = self.get_middle(channel_list)
        elif 'ignore' in self.channel_mode:
            channel_list = self.remove_channels(channel_list)
            top_channels = len(channel_list)

        channels = []
        for i in range(top_channels):
            channels.append(channel_list[i])
        return channels

    def forward(self, input):
        return self.rank_channel_list(input)



if __name__ == "__main__":
    main()