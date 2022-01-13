#!/usr/bin/env python
# coding: utf-8

# <div align=center>
# 		
# <p></p>
# <p></p>
# <font size=5>
# In the Name of God
# <font/>
# <p></p>
#  <br/>
#     <br/>
#     <br/>
# <font color=#FF7500>
# Sharif University of Technology - Departmenet of Computer Engineering
# </font>
# <p></p>
# <font color=blue>
# Artifical Intelligence - Dr. Mohammad Hossein Rohban
# </font>
# <br/>
# <br/>
# Fall 2021
# 
# </div>
# 
# <hr/>
# 		<div align=center>
# 		    <font color=red size=6>
# 			    <br />
# Practical Assignment 4 Pytorch Classification
#             	<br/>
# 			</font>
#     <br/>
#     <br/>
# <font size=4>
#                 <br/><b>
#               Cheating is Strongly Prohibited
#                 </b><br/><br/>
#                 <font color=red>
# Please run all the cells.
#      </font>
# </font>
#                 <br/>
#     </div>

# # Personal Data

# In[333]:


# Set your student number
student_number = 97101489
Name = 'Amirhosein'
Last_Name = 'Javadi'


# # Rules
# - You **are** allowed to add or remove cells. 
# - By running the cell below, you can see if your jupyter file is accepted or not. This cell will also **generate a python file which you'll have to upload to Quera** (as well as your jupyter file). The python file will later be validated and if the code in both files doesn't match, **your Practical Assignment won't be graded**.

# In[340]:


# remember to save your jupyter file before running this script
from Helper_codes.validator import *

python_code = extract_python("./Q2.ipynb")
with open(f'python_code_Q2_{student_number}.py', 'w') as file:
    file.write(python_code)


# # PyTorch & MNIST Classification (50 points+5 Extra)

# <font size=4>
# Author: Arman Zarei
# 			<br/>
#                 <font color=red>
# Please run all the cells.
#      </font>
# </font>
#                 <br/>
#     </div>

# In this assignment, you are going to learn the fundamentals of PyTorch and implement a classifier network for MNIST dataset. You can read about more details of PyTorch components from [this link](https://pytorch.org/tutorials).

# ### Setup

# In[240]:


import torch
from torchvision import datasets
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
from Helper_codes.ae_helper import init_mnist_subset_directories
import numpy as np


# In[241]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


# #### Loading dataset

# In[242]:


mnist_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())


# ## Autograd

# Autograd is PyTorch's automatic differentiation engine that powers neural network training. In essence it helps us to calculate derivatives and updating parameters. To read more about Autograd and Computational Graph visit [this link](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
# 
# Let us define two parameters: $p_1$ and $p_2$. 

# In[243]:


p1 = torch.tensor([3.], requires_grad=True)
p2 = torch.tensor([7.], requires_grad=True)


# Now define a variable $L$ as below:
# 
# $L = 3 \times p_1^3 - 7 \times p_2^2 + sin(p1) \times p_2^2$

# In[244]:


# Place your code here (2.5 Points)
## Hint: You can use torch.sin for calculation of sin(p1)
L = 3 * p1**3 - 7 * p2**2 + torch.sin(p1) * p2**2


# Now you have to calculate the gradient of $p_1$ and $p_2$ with respect to $L$. First do it by hand and write the mathematical expression in the below cell both for $p_1$ and $p_2$. Then evaluate them at $3$ and $7$ (acording to the initialization)

# Place your expressions here **(2.5 Points)**
# 
# 
# $\frac{dL}{dp_1} = 9 \times p_1^2 + cos(p_1) \times p_2^2$ 
# 
# $\frac{dL}{dp_2} = -14 \times p_2 + 2 \times sin(p1) \times p_2$
# 
# $\frac{dL}{dp_1}(3, 7) = 32.4904 $
# 
# $\frac{dL}{dp_2}(3, 7) = -96.0243$

# In[245]:


# Place your code here (optional)
## For calculation of dL/dp in the given point

# I had implemented dL/dp with Matlab code:
# syms p1;
# syms p2;
# L = @(p1,p2) 3 * p1.^3 - 7 * p2.^2 + sin(p1) * p2^2;
# f1 = diff(L ,p1, 1);
# f1 = matlabFunction(f1);
# f2 = diff(L ,p2, 1);
# f2 = matlabFunction(f2);
# f1(3,7)
# f2(3,7)


# Now let's calculate this using pytorch. The below code will construct the computational graph and stores the gradient of each variable inside of it.

# In[246]:


L.backward()


# Check whether the result of your calculations is the same with the autograd's output.

# In[247]:


print(f"P_1 grad: {p1.grad.item()}\nP_2 grad: {p2.grad.item()}")


# ## Transform 

# Data does not always come in its final processed form that is required for training machine learning algorithms. We use transforms to perform some manipulation of the data and make it suitable for training. For more details, you can read [this link](https://pytorch.org/vision/stable/transforms.html)
# 
# There are many transformations that are already implemented inside pytorch that you can use. Here we are going to implement some transformation from scratch using PyTorch's framework.

# #### Random Horizontal Flip Transformation

# Define a transformation that flips the image (horizontally) with probability of $p$. In order to implement a simple transformation class, you need to have two methods: `__init__` and `__call__` (which receives the image)

# In[248]:


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
    # Place your code here (2 Points)
        self.s = np.random.binomial(1, p, 1)
        pass
  
    def __call__(self, x):
    # Place your code here (3 Points)
        if self.s == 1 :
            im_flip = torch.flip(x,[2])
            return im_flip
        return x


# #### Random Color Swap

# Now that you have learned how to implement a transformation class, let's build another one. Define a transformation which swaps the colors (in the gray scale image) with probability of $p$. For example, if the given image had a white colored number with a black background, it should output an image with black colored number and a white background.
# 
# **Hint:** for a given input $x$, you can concider the swapped color version of it as $x_{new} = m - x$ where $m$ is the maximum value in image $x$

# In[249]:


# Place your code here (5 Points)
class RandomColorSwap(object):
    def __init__(self, p=0.5):
        self.s = np.random.binomial(1, p, 1)
        pass
    
    def __call__(self, x):
    # Place your code here (3 Points)
        if self.s == 1 :
            New_Im = 255 - x
            return New_Im
        return x


# ### Evaluation of transformations

# Now, let's apply these transformations on some images of our dataset. You can stack different transformation using `Compose`

# In[250]:


trans = transforms.Compose([
  RandomHorizontalFlip(p=0.7),
  RandomColorSwap()
])


# In[251]:


num_imgs = 10
fig, axs = plt.subplots(2, num_imgs, figsize=(25, 5))
for i, idx in enumerate(torch.randint(0, len(mnist_dataset), [num_imgs])):
    x, y = mnist_dataset[idx]
    axs[0, i].imshow(x[0], cmap='gray')
    axs[1, i].imshow(trans(x)[0], cmap='gray')
    for k in range(2):
        axs[k, i].set_yticks([])
        axs[k, i].set_xticks([])

axs[0, 0].set_ylabel("Original")
axs[1, 0].set_ylabel("Transformed");


# What we expect to see is that, some of them are fliped horizontally, some swapped in color, and some both.

# ## Dataset

# In this section, we are going to implement a dataset class. Inside `torchvision.datasets` there are many Datasets that are already implemented and ready to use. But, in many situations it's necessary to implement one by your self.

# Run the below cell to initialize our dataset

# In[252]:


dataset_path = "new_mnist"
init_mnist_subset_directories(mnist_dataset, dataset_path)


# If you take a look at `new_mnist` directory which is created in the directory you are working in, you see that it contains 10 folders with names from `0` to `9` indicating the label of the images inside it. Inside each, some files with name `data_{number}.pth` exist. Each `.pth` file is an image.
# 
# Now you have to implement a Dataset on top of these files.
# The methods that you should implement in a Dataset (as you can see in the code) is as described below:
# - `__init__`: In our example assume that it only accepts `root_dir` and `transform` as it's parameters (You should apply the transformations before outputing the data)
# - `__len__`: Should return the number of data in your dataset
# - `__getitem__`: which receives an index, should return the data at the given index (which is a tuple here, containing image and the corresponding label) 

# In[253]:


class MNISTDataset(Dataset):
    def __init__(self, root_dir, transform):
        # Place your code here (4 Points)
        ## Hint: Use os.listdir(some_path) to get the list of files
        self.list_of_files = os.listdir(root_dir)
        self.trans = transform
        pass

    def __len__(self):
        # Place your code here (1 Points)
        My_sum = 0
        for i in range(len(self.list_of_files)):
            path = "new_mnist/" + self.list_of_files[i]
            My_list = os.listdir(path)
            My_sum += len(My_list)
        return My_sum

    def __getitem__(self, idx):
        query = "data_" + str(idx) + ".pth"
        for i in range(len(self.list_of_files)):
            PATH1 = "new_mnist/" + self.list_of_files[i]
            My_list = os.listdir(PATH1)
            if query in My_list:
                PATH2 = "new_mnist/" + str(i) + "/" + query
                file = torch.load(PATH2)
                return (i,file)
            
        # Place your code here (3 Points)
        ## Return a tuple (image, label)


# After you defined your dataset, let's use it.

# In[254]:


my_dataset = MNISTDataset(root_dir=dataset_path, transform=RandomColorSwap())
len(my_dataset)


# Visualize 10 samples (randomely) from your dataset with their labels

# In[255]:


# Place your code here (2 Points)
n = len(my_dataset)
k = 10
random_samples = np.random.randint(n, size=k)
fig, axs = plt.subplots(2, k//2, figsize=(25, 7))
for i in range(k//2):
    t = my_dataset[random_samples[i]]
    image = np.reshape(t[1].numpy(),[28,28])
    axs[0, i].imshow(image, cmap='gray')
    axs[0, i].set_title(f'Label = {t[0]}')
    t = my_dataset[random_samples[i+k//2]]
    image = np.reshape(t[1].numpy(),[28,28])
    axs[1, i].imshow(image, cmap='gray')
    axs[1, i].set_title(f'Label = {t[0]}')


# ## MNIST Classification

# ### Model
# Define your model (Based on what you have learned in the workshop). I highly encourage you to try different models with different layers in order to achieve a better accuracy
# 
# **Notice:** You cannot use convolution layers in your model 

# In[274]:


# Place your code here (5 points)
class DigitRecognizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(28*28, 256)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(256, 10)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B  I
        x = self.flatten(x)
        x = self.lin1(x)    # B 256
        x = self.relu(x)
        x = self.lin2(x)    # B C
        return x


# In[275]:


model = DigitRecognizer().to(device)
model


# ### Dataset and Dataloader

# Prepare datasets and dataloader for train/test. We use MNIST dataset that is already implemented inside `torchvision.datasets`.
# 
#  You need to split the `mnist_dataset` into `train_dataset` and `val_dataset`. 
#  
#  You can also define more transformations that you think it would help the training process **(Optional)**.

# In[276]:


transform_compose = transforms.Compose([
    transforms.ToTensor(),
    # Place your code here
])

mnist_dataset = datasets.MNIST(root='dataset', train=True, download=True, transform=transform_compose)
train_size = int(0.9 * len(mnist_dataset))
val_size = len(mnist_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(mnist_dataset, (train_size, val_size))

# Place your code here (1 points)
## Notice: Split mnist_dataset into train_dataset and val_dataset

# End of block for your code placement
test_dataset = datasets.MNIST(root='dataset', train=False, download=True, transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)


# ### Criterion and Optimizer

# you have to tune the `learning_rate` yourself (Based on your training process)

# In[277]:


criterion = nn.CrossEntropyLoss()
learning_rate= 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# ### Train your model

# Write your training/validation loop in the following cell for an arbitrary number of epochs (until convergence is detected). You also need to store train/validation loss at each epoch in order to visualize them the after training is done.

# In[329]:


import tqdm
num_epochs= 10
batch_size = 32
train_loss_arr, val_loss_arr = [], []
for epoch in range(num_epochs):
    train_loss, val_loss = 0, 0
    for step, (x, y) in enumerate(train_loader): 
            #print(x.size())
            #print(y)
            x = x.to(device)
            y = y.to(device)
            p = model(x)
            loss = criterion(p, y)
            train_loss += float(loss) * batch_size
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    #model.train()
    # Place your code here (5 points)
    ## Hint: Loop throught train_loader, get images with their labels and train the model
    ## Hint 2: To update train_loss inside the loop use "train_loss += batch_loss * batch_size""
    
    for step, (x, y) in enumerate(val_loader): 
            x = x.to(device)
            y = y.to(device)
            p = model(x)
            loss = criterion(p, y)
            val_loss += float(loss) * batch_size
         
    # model.eval()
    # Place your code here (4 points)
    ## Hint: Loop throught val_loader, get images with their labels and evaluate the model
    ## Hint 2: To update val_loss inside the loop use "val_loss += batch_loss * batch_size""


    train_loss /= len(train_loader.dataset)
    val_loss /= len(val_loader.dataset)
    train_loss_arr.append(train_loss)
    val_loss_arr.append(val_loss)

    print(f"[Epoch {epoch}]\t"
        f"Train Loss: {train_loss:.4f}\t"
        f"Validation Loss: {val_loss:.4f}")


# #### Plot train/validation loss

# In[330]:


plt.plot(train_loss_arr)
plt.plot(val_loss_arr)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train loss', 'validation loss'])
plt.grid()
plt.show()


# ### Evaluation

# Evaluate your model on test set. you have to report both loss and accuracy.
# 
# In order to get the full score of this section, you need to achieve an accuracy above $95\%$ and to get the extra points, your accuracy should be above $98\%$ 
# 
# **Notice:** You should use test set only for evaluation of your model.

# In[337]:


test_loss = 0
epoch_all = 0
epoch_true = 0
for step, (x, y) in enumerate(test_loader): 
    x = x.to(device)
    y = y.to(device)
    p = model(x)
    loss = criterion(p, y)
    test_loss += float(loss) * batch_size    
    predictions = p.argmax(-1)
    epoch_all += len(predictions)
    epoch_true += (predictions == y).sum()
    
test_loss /= len(test_loader.dataset)
print(f'Test Loss: {test_loss :.4e} - Acc: {epoch_true * 100. / epoch_all:.4f}%')  

Test Loss: 8.1477e-02 - Acc: 98.0600%
# ### Visualization of wrong prediction

# Visualize $8$ samples (along with original label and model's prediction) from test set which are labeled wrong by your model

# In[332]:


# Place your code here (5 points)
fig, axs = plt.subplots(2, 4, figsize=(25, 7))
counter = 0
Flag = True
for step, (x, y) in enumerate(test_loader): 
    if Flag == False:
        break
    x = x.to(device)
    y = y.to(device)
    p = model(x)
    predictions = p.argmax(-1)
    n = int(predictions.size(dim=0))
    for i in range(n):
        if predictions[i] != y[i]:
            image = np.reshape(x[i].numpy(),[28,28])
            if counter<4:
                axs[0, counter].imshow(image, cmap='gray')
                counter += 1
            elif counter<8:
                axs[1, counter-4].imshow(image, cmap='gray')
                counter += 1
            else:
                Flag = False
                break
    

