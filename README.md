# AutoPrune: Pruning Neural Networks using Deep Reinforcement Learning

We compress our network using Weight Pruning. We use a Reinforcement Learning (RL) based pruning algorithm, the AutoPrune algorithm, to prune a pre-trained Neural Network. Specifically, we prune each layer of a target network using a compression rate given to us by the RL agent. Our AutoPrune algorithm co-optimizes for a target accuracy and a target sparsity ratio. We do fine-tuning for a few epochs once the pruning process finishes. 

**We achieve a score of xx on CIFAR100 and xx on ImageNet**

## Our Algorithm
We formulate pruning a Neural Network as a Markov Decision Process (MDP). Our AutoPrune algorithm maintains a representation of the network being pruned, which is called the ‘state’. For each layer of the network to be pruned or the target network, AutoPrune gives us a compression rate α by which we prune that layer. Once, the layer is pruned, a reward is returned back to AutoPrune, to let it know whether the compression rate was good or not. Processing all the layers in the target network in this way, is called an episode. AutoPrune trains itself on a specified number of episodes, and learns a unique compression ratio for each layer in the network in the end. 

We use Deep Q-learning as our underlying RL algorithm due to its fast learning speed. Our method is different from past RL based pruning algorithms, notably AMC by (He et al., 2018) in many aspects, one of them being by giving dense rewards to the agent rather than sparse rewards. Our reward function comprises of two terms, a sparsity penalty and an accuracy penalty, where a baseline sparsity target and baseline accuracy target is set by the user. 

*Reward(R) ~= Min(Current Accuracy – Target Accuracy, 0) + Min(Current Sparsity – Target Sparsity, 0)*

In this way, AutoPrune co-optimizes for sparsity and accuracy at each layer. The compression rate α given by AutoPrune does magnitude pruning on the network (Han et al., 2015). Specifically, the magnitude threshold for pruning each layer is calculated as-

*Pruning Threshold = αt σ(wt)*

Where αt is the alpha for a given layer t in the Target Network and σ(wt) is the standard deviation of the weights for the layer t. This prunes all the connections in the layer which are below the magnitude threshold. We search for α over a given range of discrete standard deviation values i.e. α ∈ {1.0, 1.2, ···, 2.2}.

We do retraining after pruning each layer. Once, an episode finishes, the original model is loaded again to start the next episode. Once, all the episodes are finished, the final compression rates per layer are obtained. We then prune the Target Network with these compression rates and finetune it to report the final accuracy of our pruned model.

Full paper to be released soon.

## FLOPs Calculation
For calculating the number of parameters in our network, we follow the guidelines provided. I.e. we count the total number of parameters remaining after pruning and use a 16bit size (as per the freebie quantization guidelines given) to calculate the size of the parameters. We also add on the size of the weight mask used for pruning. For the weight mask size we take all the parameters in the starting model and multiply it by 1bit (the size to denote mask information per parameter). We then add the weight mask size and the size of remaining parameters to get the total number of parameters in our model.

For flops, we calculate flops on a per layer basis. We calculate flops on the remaining weights after pruning. We calculate multiply operations as 16bit and add operations as 32bit. We then add together the multiply and adds per layer and do it for each layer. Adding together the flops for each layer gives us the total number of flops for the whole model. 

**We achieve xx Params and xx Flops on CIFAR100 leading to a score of xx**

**We achieve xx Params and xx Flops on ImageNet leading to a score of xx**

We use WideResnet-28-10 as our starting network for CIFAR100 and EfficientNet-B2 for ImageNet. We include checkpoints of the pruned models for reference. We also include testing scripts to test the validation accuracy of the included checkpoints. The instruction for running the script are provided below.

## Running the script 
Evaluate WideResNet-28-10 for CIFAR100-

```
python test.py --wideresnet -e --resume /path/to/checkpoint.pth --batch-size 128 /path/to/data/directory 
```
Evaluate EfficientNet-B2 for ImageNet-

```
python test.py --wideresnet -e --resume /path/to/checkpoint.pth --batch-size 128 /path/to/data/directory 
```
